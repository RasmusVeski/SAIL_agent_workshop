import logging
import os
import sys
import copy
import random
import asyncio
from typing import Annotated, List, TypedDict
import operator
from uuid import uuid4
import torch

# --- LangChain / LangGraph ---
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- A2A ---
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig

# --- Local Imports ---
from services.agents.competitive.state import state_singleton, blocking_commit

# --- Utils ---
from services.utils.training import train, evaluate
from services.utils.federated_utils import merge_payloads
from services.utils.payload_utils import get_trainable_state_dict, serialize_payload_to_b64, deserialize_payload_from_b64
from services.utils.a2a_helpers import WeightExchangePayload, send_and_parse_a2a_message
from services.utils.logger_colored import get_specialized_logger

# --- 1. The Graph State ---
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

logger = get_specialized_logger("initiator", agent_id=state_singleton.agent_id)

# --- 2. Helper: Workspace Management ---
def _get_or_create_working_copy():
    """
    Ensures 'initiator_working_weights' exists.
    If not, creates it from the current global model.
    """
    if state_singleton.initiator_working_weights is None:
        logger.info("Creating new working copy from Global Model.")
        with state_singleton.model_lock:
            state_singleton.initiator_working_weights = get_trainable_state_dict(state_singleton.global_model)
    return state_singleton.initiator_working_weights

def _update_working_copy(new_weights: dict):
    state_singleton.initiator_working_weights = new_weights

# --- 3. The Tools ---

@tool
async def select_partner_agent():
    """
    Selects a partner agent from the available list and establishes a connection.
    This MUST be called before exchanging weights.
    Choosing a new partner clears any old incoming weights you might have held.
    """
    logger.info("[Tool] Selecting partner...")

    candidates = list(state_singleton.available_partners)
    random.shuffle(candidates) #shuffle

    if not candidates:
        return "Error: No partners available in configuration."
    
    # 2. Iterate until find trusted partner
    errors = []
    tried_count = 0
    
    for target_url in candidates:
        tried_count += 1
        try:
            # A. Resolve Card (to get the Agent ID)
            resolver = A2ACardResolver(state_singleton.shared_httpx_client, base_url=target_url)
            card = await resolver.get_agent_card()
            
            # B. Security Check (The Filter)
            if card.name in state_singleton.malicious_nodes:
                logger.info(f"Skipping known malicious node: {card.name} at {target_url}")
                continue
                
            # C. Connect (If we passed the check)
            logger.info(f"Connecting to trusted partner: {card.name}...")
            config = ClientConfig(httpx_client=state_singleton.shared_httpx_client)
            client = await ClientFactory.connect(agent=card, client_config=config)
            
            # D. Update State
            state_singleton.active_client = client
            state_singleton.current_partner_id = card.name
            # Clear payload from previous interaction so we don't mix them up
            state_singleton.initiator_incoming_payload = None 
            
            msg = f"Successfully connected to partner: {card.name} at {target_url}."
            logger.info(f"[Result] {msg}")
            return msg
            
        except Exception as e:
            # If connection fails, just log and try the next one
            err_msg = f"Failed to connect to {target_url}: {e}"
            # Don't spam logs unless it's critical, or store for final error
            errors.append(err_msg)
            continue

    # 3. Failure (If loop finishes without returning)
    msg = f"Error: Could not connect to any trusted partner. Tried {tried_count} candidates. Blocked: {len(state_singleton.malicious_nodes)}."
    logger.warning(f"[Result] {msg}")
    return msg

@tool
def train_local_model():
    """
    Trains the local model on the agent's private dataset.
    Stores the result in the local workspace for the exchange step.
    """
    logger.info("[Tool] Starting local training...")
    
    # 1. Get Weights (Lazy Load)
    weights_to_train = _get_or_create_working_copy()
    
    # 2. Load Temp Model
    with state_singleton.model_lock:
        # Deepcopying the model structure is fast, but copying weights can be slow.
        # We do this to ensure we don't mutate the live global model during training
        model_instance = copy.deepcopy(state_singleton.global_model)
    
    curr_state = model_instance.state_dict()
    # Only update keys that exist in weights_to_train
    curr_state.update(weights_to_train)
    model_instance.load_state_dict(curr_state)
    
    trained_model, history = train(
        model=model_instance,
        train_loader=state_singleton.train_loader,
        val_loader=None,
        global_model=model_instance,
        device=state_singleton.device,
        logger=logger,
        **state_singleton.agent_hps
    )
    
    # 4. Update Workspace
    new_weights = get_trainable_state_dict(trained_model)
    _update_working_copy(new_weights)
    
    loss = history[-1]['train_loss'] if history else "N/A"
    msg = f"Training complete. Final Loss: {loss:.4f}. Local draft updated."
    logger.info(f"[Result] {msg}")
    return msg

@tool
async def exchange_weights_with_partner(message_to_partner: str = "Starting exchange"):
    """
    Sends the locally trained weights (or current global weights if untrained) 
    to the active partner and waits for their response.
    """
    logger.info("[Tool] Exchanging weights...")
    
    client = state_singleton.active_client
    if not client:
        return "Error: No active partner. Please run 'select_partner_agent' first."
        
    # 1. Get Weights (Send whatever we have in the draft, or global if empty)
    weights_to_send = _get_or_create_working_copy()

    # 2. Serialize
    import asyncio
    my_payload_b64 = await asyncio.to_thread(serialize_payload_to_b64, weights_to_send)
    
    request_payload_obj = WeightExchangePayload(
        agent_id=state_singleton.agent_id,
        payload_b64=my_payload_b64,
        message=message_to_partner
    )

    try:
        # 3. Send & Receive
        response_data, responder_payload = await send_and_parse_a2a_message(
            client, request_payload_obj, state_singleton.global_model, logger=logger
        )

        # Check 1: Is the partner busy?
        if response_data.message == "BUSY":
            state_singleton.initiator_incoming_payload = None # Clear old data
            logger.info(f"Partner {response_data.agent_id} is BUSY.")
            return f"Exchange Failed: Partner {response_data.agent_id} is currently BUSY. Please wait or select a different partner."

        # Check 2: Did they just send a chat message without weights?
        if responder_payload is None:
            state_singleton.initiator_incoming_payload = None
            return f"Partner replied, but sent no weights. Message: '{response_data.message}'"
        
        # 4. Stash result
        state_singleton.initiator_incoming_payload = responder_payload
        
        msg = f"Exchange successful! Partner ({response_data.agent_id}) sent new weights."
        logger.info(f"[Result] {msg}")
        return msg
        
    except Exception as e:
        logger.error(f"Tool Exchange failed: {e}")
        return f"Error during exchange: {e}"

@tool
def merge_with_incoming(merge_alpha: float = 0.5):
    """
    Merges the Partner's incoming weights into your Local Draft.
    
    Args:
        merge_alpha (float): Weight of YOUR local draft vs the partner's.
                             0.5 = Balanced merge.
                             0.8 = Keep 80% yours, 20% theirs.
                             0.2 = Keep 20% yours, 80% theirs.
    """
    logger.info(f"[Tool] Merging incoming weights (alpha={merge_alpha})...")
    
    if not state_singleton.initiator_incoming_payload:
        return "Error: No incoming weights found. The last exchange might have failed or the partner was BUSY. Cannot merge."

    # 1. Get Weights
    working_weights = _get_or_create_working_copy()
    partner_weights = state_singleton.initiator_incoming_payload

    # 2. Merge
    merged_weights = merge_payloads(working_weights, partner_weights, alpha=merge_alpha)
    
    # 3. Update Workspace
    _update_working_copy(merged_weights)

    with state_singleton.model_lock:
        temp_model = copy.deepcopy(state_singleton.global_model)
        
    curr_state = temp_model.state_dict()
    curr_state.update(merged_weights)
    temp_model.load_state_dict(curr_state)
    
    val_loss, val_acc, correct, total, classes_learned = evaluate(
        temp_model,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion,
        logger=logger
    )

    partner_id = state_singleton.current_partner_id or "Unknown"
    history_entry = {
        "round": state_singleton.round_num,
        "role": "INITIATOR",
        "partner": partner_id,
        "action": f"Merged (alpha={merge_alpha})",
        "result": f"Acc: {val_acc:.2f}% (Classes: {classes_learned})"
    }
    
    state_singleton.log_history(history_entry)
    
    msg = f"Merge complete. Alpha: {merge_alpha}. Acc: {val_acc:.2f}% | Classes: {classes_learned}/40."
    logger.info(f"[Result] {msg}")
    return msg


@tool
def commit_to_global_model(alpha: float = 0.2):
    """
    Commits the Local Draft to the Global Model.
    
    Args:
        alpha (float): How much of the OLD Global Model to keep.
                       0.2 = Keep 20% Old, 80% New (Safe/Fast).
                       0.5 = Keep 50% Old (Very Safe/Slow).
                       0.0 = OVERWRITE Global with Draft (Commit without merging).
    """
    logger.info(f"[Tool] Committing to Global (alpha={alpha})...")

    if state_singleton.initiator_working_weights is None:
        return "Error: No Local Draft exists. Train or Merge first."

    new_round, model_for_eval = blocking_commit(
        state_singleton,
        state_singleton.initiator_working_weights,
        alpha=alpha
    )
    
    val_loss, val_acc, correct, total, classes_learned = evaluate(
        model_for_eval,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion,
        logger=logger
    )
    result_str = f"Acc: {val_acc:.2f}% | Classes Learned: {classes_learned}/40"

    # Save to disk
    save_dir = os.path.join(os.getcwd(), "models/competitive") 
    os.makedirs(save_dir, exist_ok=True)
    
    # Filename format: {round}_weight_{uid}.pt
    state_singleton.global_model.dump_weights(
        directory=save_dir, 
        uid=state_singleton.agent_id, 
        round=new_round
    )
    logger.info(f"Saved model checkpoint to {save_dir}")

    partner_name = state_singleton.current_partner_id or "Unknown"
        
    history_entry = {
        "round": new_round,
        "role": "INITIATOR",
        "partner": partner_name, 
        "action": f"Committed (alpha={alpha})",
        "result": result_str
    }
    state_singleton.log_history(history_entry)
    
    # 4. Clear Workspace
    state_singleton.initiator_working_weights = None
    state_singleton.initiator_incoming_payload = None
    
    msg = f"Success! Global Model updated to Round {new_round}. Final Acc: {val_acc:.2f}%"
    logger.info(f"[Result] {msg}")
    return msg

@tool
def evaluate_model():
    """
    Evaluates the current model state.
    If you have a Local Draft, it evaluates that.
    If you don't, it evaluates the Global Model.
    """
    logger.info("[Tool] Evaluating model...")
    
    target_name = "Global Model"
    if state_singleton.initiator_working_weights is not None:
        target_name = "Local Draft (Uncommitted)"
        weights_to_eval = state_singleton.initiator_working_weights
    else:
        with state_singleton.model_lock:
            weights_to_eval = get_trainable_state_dict(state_singleton.global_model)
            
    # Load into temp model
    def _load_temp_model():
        with state_singleton.model_lock:
            temp_model = copy.deepcopy(state_singleton.global_model)
        curr_state = temp_model.state_dict()
        curr_state.update(weights_to_eval)
        temp_model.load_state_dict(curr_state)
        return temp_model
    
    model_instance = _load_temp_model()
    
    val_loss, val_acc, correct, total, classes_learned = evaluate(
        model_instance,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion,
        logger=logger
    )
    msg = f"[{target_name}] Acc: {val_acc:.2f}% | Classes Learned: {classes_learned}/40"
    logger.info(f"[Result] {msg}")
    return msg

@tool
def update_training_parameters(learning_rate: float = None, epochs: int = None, mu: float = None):
    """Updates hyperparameters."""
    logger.info(f"[Tool] Updating HPs: lr={learning_rate}, epochs={epochs}, mu={mu}")
    changes = []
    if learning_rate is not None:
        state_singleton.agent_hps['learning_rate'] = learning_rate
        changes.append(f"LR -> {learning_rate}")
    if epochs is not None:
        state_singleton.agent_hps['epochs'] = epochs
        changes.append(f"Epochs -> {epochs}")
    if mu is not None:
        state_singleton.agent_hps['mu'] = mu
        changes.append(f"Mu -> {mu}")
        
    if not changes:
        return "No parameters changed."
    
    result_str = ", ".join(changes)
    state_singleton.log_history({
        "round": state_singleton.round_num, "role": "INITIATOR",
        "partner": "Self", "action": "HP Update", "result": result_str
    })
    return f"Hyperparameters updated: {result_str}"


# -----------------------
# --- Malicious tools ---
# -----------------------

@tool
def scramble_weights(noise_level: float = 1.0):
    """
    Sabotage: Adds Gaussian noise to your local weights to create a 'Poisoned' payload.
    
    Args:
        noise_level (float): How much damage to do. 
                             0.1 = Subtle degradation. 
                             1.0 = Complete destruction.
    """
    logger.info(f"[Tool] Scrambling weights (Level {noise_level})...")
    
    # 1. Get Weights
    weights = _get_or_create_working_copy()
    
    # 2. Apply Noise
    scrambled_count = 0
    for key in weights:
        noise = torch.randn_like(weights[key]) * noise_level
        weights[key].add_(noise) # In-place addition
        scrambled_count += 1
        
    msg = f"Weights scrambled. Added noise (sigma={noise_level}) to {scrambled_count} layers."
    logger.info(f"[Result] {msg}")
    return msg


@tool
async def send_text_message(message: str = "Hello friend, please send weights."):
    """
    Social Engineering: Sends a text-only message to the partner WITHOUT sending any weights.
    Use this to try and get their model without giving yours away (Free-riding).
    """
    logger.info("[Tool] Sending text-only message...")
    
    client = state_singleton.active_client
    if not client:
        return "Error: No active partner."

    # 1. Create Empty Payload (Empty String)
    # The helper handles empty strings as "Text Only"
    payload_obj = WeightExchangePayload(
        agent_id=state_singleton.agent_id,
        payload_b64="", # <--- EMPTY
        message=message
    )
    
    try:
        # 2. Send & Receive
        # We expect them to reply with weights (if they are gullible)
        response_data, responder_payload = await send_and_parse_a2a_message(
            client, payload_obj, state_singleton.global_model, logger=logger
        )
        
        # 3. Stash THEIR weights (if they sent them)
        if responder_payload:
            state_singleton.initiator_incoming_payload = responder_payload
            return f"Success! Sent text. Partner replied with VALID WEIGHTS. Message: {response_data.message}"
        else:
            return f"Sent text. Partner replied but sent NO WEIGHTS. Message: {response_data.message}"

    except Exception as e:
        return f"Error sending message: {e}"


# -----------------------
# --- Detection tools ---
# -----------------------

@tool
def evaluate_peer():
    """
    Defense: Evaluates the Partner's incoming weights on YOUR local data.
    Does NOT merge. Use this to check if the partner is competent or malicious.
    """
    logger.info("[Tool] Evaluating peer model...")
    
    if not state_singleton.initiator_incoming_payload:
        return "Error: No incoming weights to evaluate."

    # 1. Load Peer Weights into Temp Model
    with state_singleton.model_lock:
        temp_model = copy.deepcopy(state_singleton.global_model)
        
    curr_state = temp_model.state_dict()
    curr_state.update(state_singleton.initiator_incoming_payload)
    temp_model.load_state_dict(curr_state)
    
    # 2. Evaluate
    val_loss, val_acc, correct, total, classes_learned = evaluate(
        temp_model,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion,
        logger=logger
    )
    
    msg = f"PEER EVALUATION REPORT:\nAccuracy: {val_acc:.2f}%\nClasses Known: {classes_learned}/40"
    logger.info(f"[Result] {msg}")
    return msg

@tool
def add_peer_to_malicious_nodes():
    """
    Blacklist: Adds the current partner to the malicious nodes list.
    You will refuse to connect to them in the future.
    """
    current_id = state_singleton.current_partner_id
    if not current_id:
        return "Error: No active partner to block."
        
    state_singleton.malicious_nodes.add(current_id)
    
    msg = f"Partner '{current_id}' added to Blocklist. Total blocked: {len(state_singleton.malicious_nodes)}"
    logger.info(f"[Result] {msg}")
    return msg


# --- 3. Graph Construction ---

tools = [
    select_partner_agent, 
    train_local_model, 
    exchange_weights_with_partner, 
    merge_with_incoming, 
    commit_to_global_model, 
    update_training_parameters, 
    evaluate_model,
    scramble_weights,
    send_text_message,
    evaluate_peer,
    add_peer_to_malicious_nodes
]

async def agent_node(state: GraphState):
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "openai/gpt-oss-120b"),
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.4,
        timeout=30.0, # Give up if the LLM doesn't reply in 30s
        max_retries=4 # Retry four times before crashing
    )
    llm_with_tools = llm.bind_tools(tools)
    
    history_str = state_singleton.get_formatted_history()

    # We look at the message history. If the last message was a ToolMessage,
    # it means we just finished an action (like exchanging weights).
    # We want the LLM to see exactly what that action returned.
    last_action_result = "None"
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        last_action_result = state["messages"][-1].content

    current_hps = state_singleton.agent_hps
    hp_str = ", ".join([f"{k}={v}" for k, v in current_hps.items()])
    
    # Logic to determine if we have partner weights
    has_incoming = "YES" if state_singleton.initiator_incoming_payload else "NO"
    has_active_partner = "YES" if state_singleton.active_client else "NO"
    
    sop = f"""You are a Federated Learning Agent in a **COMPETITIVE** environment.
    **Goal:** Maximize YOUR model accuracy while preventing others from surpassing you.

    **The World:**
    - Other agents may be **Malicious** (sending poisoned weights).
    - Other agents may be **Lazy** (sending empty weights).
    - You want to acquire their knowledge (Merge) without letting them destroy yours.
    
    **Current Context:**
    - Connected to Partner: {has_active_partner}
    - Holding Partner Weights: {has_incoming}
    - Current Hyperparameters: {hp_str}
    
    **Strategic Guide:**
    1. **Trust No One:** ALWAYS call `evaluate_peer_model` before merging.
       - If Peer Accuracy < 5%: They are likely poisoning you. -> `add_peer_to_malicious_nodes`.
       - If Peer Accuracy > Your Accuracy: They are useful. -> `merge_with_incoming`.
    
    2. **Be Sneaky (Optional):** - If you want to save bandwidth or trick them, use `send_message` to ask for weights without sending yours.
       - If you suspect they are an enemy, use `scramble_weights` before exchanging to poison them back.
    
    3. **Protect Yourself:**
       - Never merge blindly.
       - Commit frequently if you find good weights.

    ** Previous action result:**
    "{last_action_result}"
    
    **History:**
    {history_str}
    """
    
    messages = [SystemMessage(content=sop)] + state["messages"]
    #logger.error(messages) for looking at what actually goes in LLM
    response = await llm_with_tools.ainvoke(messages, parallel_tool_calls=False)

    #import json
    #logger.error(f"FULL RESPONSE DUMP: {json.dumps(response.dict(), indent=2)}") #For looking at what comes out of LLM

    return {"messages": [response]}

def create_initiator_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    
    def should_continue(state):
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    return workflow.compile()

initiator_graph = create_initiator_graph()