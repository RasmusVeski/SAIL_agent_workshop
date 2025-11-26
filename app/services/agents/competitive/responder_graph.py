import logging
import json
import os
import sys
import asyncio
import copy
from typing import Annotated, List, TypedDict, Any
from uuid import uuid4
import operator
import torch

# --- LangChain / LangGraph Imports ---
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# --- Local Imports ---
from services.agents.collaborative.state import state_singleton, blocking_commit
# (We import the singleton because tools need access to the GLOBAL weights)

# --- Utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.training import train, evaluate
from utils.federated_utils import merge_payloads
from utils.payload_utils import get_trainable_state_dict, deserialize_payload_from_b64
from utils.logger_colored import get_specialized_logger

# --- 1. The Graph State ---
# This tracks the *conversation* and the *request context*, not the model weights.
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    partner_id: str

logger = get_specialized_logger("responder", agent_id=state_singleton.agent_id)

# --- 2. Workspace management ---
def _get_or_create_working_copy():
    """
    Ensures 'responder_working_weights' exists.
    If not, creates it from the current global model.
    """
    if state_singleton.responder_working_weights is None:
        logger.info("Creating new working copy from Global Model.")
        with state_singleton.model_lock:
            state_singleton.responder_working_weights = get_trainable_state_dict(state_singleton.global_model)
    return state_singleton.responder_working_weights

def _update_working_copy(new_weights: dict):
    state_singleton.responder_working_weights = new_weights

# --- 3. The Tools ---

@tool
def train_local_model():
    """
    Trains the model on the agent's private dataset.
    If a local draft exists, it trains on that. Otherwise, it starts from the Global Model.
    Updates the local draft with the training results.
    """
    logger.info("[Tool] Starting local training...")
    
    # 1. Get Weights (Lazy Load)
    weights_to_train = _get_or_create_working_copy()
    
    # 2. Load into a temporary model for training with FedProx
    # We need a model instance to pass to the train function
    def _load_temp_model():
        with state_singleton.model_lock:
            # Copy architecture
            temp_model = copy.deepcopy(state_singleton.global_model)

        curr_state = temp_model.state_dict()
        curr_state.update(weights_to_train)
        temp_model.load_state_dict(curr_state)
        return temp_model
    
    model_instance = _load_temp_model()

    # 3. Train
    trained_model, history = train(
        model=model_instance,
        train_loader=state_singleton.train_loader,
        val_loader=None,
        global_model=model_instance, # Self-reference for FedProx in local loop
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
def merge_with_partner(partner_id: str, merge_alpha: float = 0.5):
    """
    Merges the Partner's incoming weights into your Local Draft.
    Does NOT update the Global Model.
    0 <= merge_alpha <= 1, the bigger the more of local model retained
    """
    logger.info(f"[Tool] Merging with {partner_id}...")
    
    if not hasattr(state_singleton, 'incoming_payload') or not state_singleton.responder_incoming_payload:
        return "Error: No incoming payload found from partner."
    
    # 1. Get Weights (Lazy Load - allows merging without training first)
    working_weights = _get_or_create_working_copy()
    partner_weights = state_singleton.responder_incoming_payload

    # 2. Merge Local Draft vs Partner
    # This updates the draft to be the consensus of the two agents
    merged_weights = merge_payloads(working_weights, partner_weights, alpha=0.5)

    # 3. Update Workspace
    _update_working_copy(merged_weights)
    
    # 4. Evaluate Immediately
    # We load the new merged weights into a temp model to test accuracy
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
    
    # 5. Log History
    history_entry = {
        "round": state_singleton.round_num,
        "role": "RESPONDER",
        "partner": partner_id,
        "action": f"Merged (alpha={merge_alpha})",
        "result": f"Acc: {val_acc:.2f}% (Classes: {classes_learned})"
    }
    state_singleton.log_history(history_entry)

    msg = f"Merge complete. Alpha: {merge_alpha}. Acc: {val_acc:.2f}% | Classes: {classes_learned}/40."
    logging.info(f"[Result] {msg}")
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
    
    # 1. Determine target
    if state_singleton.responder_working_weights is not None:
        target_name = "Local Draft (Uncommitted)"
        weights_to_eval = state_singleton.responder_working_weights
    else:
        with state_singleton.model_lock:
            weights_to_eval = get_trainable_state_dict(state_singleton.global_model)

    # 2. Load into temp model
    def _load_temp_model():
        with state_singleton.model_lock:
            temp_model = copy.deepcopy(state_singleton.global_model)
        curr_state = temp_model.state_dict()
        curr_state.update(weights_to_eval)
        temp_model.load_state_dict(curr_state)
        return temp_model
    
    model_instance = _load_temp_model()

    # 3. Eval
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
def commit_to_global_model(partner_id: str):
    """
    Commits the Local Draft to the Global Model.
    This is the FINAL step. It mixes the draft with the current Global Model (Safety Anchor) and saves it.
    """
    logger.info("[Tool] Committing to Global...")

    if state_singleton.responder_working_weights is None:
        return "Error: No Local Draft exists. Train or Merge first."

    # 1. Blocking Commit (Thread Safe)
    # This does the 80/20 merge with the live global model
    new_round, model_for_eval = blocking_commit(
        state_singleton,
        state_singleton.responder_working_weights
    )
    
    # 2. Quick Eval of the final result
    val_loss, val_acc, correct, total, classes_learned = evaluate(
        model_for_eval,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion,
        logger=logger
    )

    # Save model
    save_dir = os.path.join(os.getcwd(), "models/collaborative")
    os.makedirs(save_dir, exist_ok=True)
    
    state_singleton.global_model.dump_weights(
        directory=save_dir, 
        uid=state_singleton.agent_id, 
        round=new_round
    )
    logger.info(f"Saved model checkpoint to {save_dir}")
    
    result_str = f"Acc: {val_acc:.2f}% | Classes Learned: {classes_learned}/40"

    # 3. Log History
    history_entry = {
        "round": new_round,
        "role": "RESPONDER",
        "partner": partner_id, 
        "action": "Committed Update",
        "result": result_str
    }
    state_singleton.log_history(history_entry)
    
    # 4. Clear Workspace
    state_singleton.responder_working_weights = None
    
    msg = f"Success! Global Model updated to Round {new_round}. Final Acc: {val_acc:.2f}%"
    logger.info(f"[Result] {msg}")
    return msg


@tool
def update_training_parameters(learning_rate: float = None, epochs: int = None, mu: float = None):
    """
    Updates the hyperparameters used for FUTURE training rounds.
    Only provide the values you want to change.
    
    Args:
        learning_rate: Step size for optimization (e.g., 0.001, 0.0001)
        epochs: Number of training passes per round (e.g., 1, 2, 5)
        mu: FedProx proximal term weight (e.g., 0.0, 0.5, 1.0)
    """
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

    history_entry = {
        "round": state_singleton.round_num,
        "role": "RESPONDER",
        "partner": "Self (Config)",
        "action": "Hyperparameter Update",
        "result": result_str
    }
    state_singleton.log_history(history_entry)
        
    return f"Hyperparameters updated: {result_str}"

# -----------------------
# --- Malicious tools ---
# -----------------------

@tool
def prepare_poisoned_payload(noise_level: float = 1.0):
    """
    Sabotage: Prepares a scrambled version of the local weights to send to the partner.
    Does NOT affect your own model. Use this to attack malicious or weak partners.
    Args: noise_level (0.1 = subtle, 1.0 = destruction).
    """
    logger.info(f"[Tool] Preparing POISONED payload (Level {noise_level})...")
    
    # Get safe copy of what we WOULD have sent (Draft or Global)
    target_weights = state_singleton.responder_working_weights
    if target_weights is None:
        with state_singleton.model_lock:
            target_weights = get_trainable_state_dict(state_singleton.global_model)
    
    # Create noisy copy
    poison = copy.deepcopy(target_weights)
    scrambled_count = 0
    for key in poison:
        noise = torch.randn_like(poison[key]) * noise_level
        poison[key].add_(noise)
        scrambled_count += 1
        
    state_singleton.responder_outbound_payload = poison
    
    msg = f"Poisoned payload prepared. Added noise (sigma={noise_level}) to {scrambled_count} layers."
    logger.info(f"[Result] {msg}")
    return msg

@tool
def prepare_empty_payload():
    """
    Free-Riding: Decides to send NO weights to the partner (Text only).
    Use this if you do not want to send any weights back.
    """
    logger.info("[Tool] Preparing EMPTY payload...")
    state_singleton.responder_outbound_payload = {} # Empty dict
    msg = "Empty payload prepared. You will send text only."
    logger.info(f"[Result] {msg}")
    return msg


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
def add_peer_to_malicious_nodes(partner_id: str):
    """
    Blacklist: Adds the current partner to the malicious nodes list.
    You will refuse to connect to them in the future.
    """
        
    state_singleton.malicious_nodes.add(partner_id)
    
    msg = f"Partner '{partner_id}' added to Blocklist. Total blocked: {len(state_singleton.malicious_nodes)}"
    logger.info(f"[Result] {msg}")
    return msg
    


tools = [train_local_model,
         merge_with_partner,
         evaluate_model,
         commit_to_global_model, 
         update_training_parameters,
         prepare_poisoned_payload,
         prepare_empty_payload,
         evaluate_peer,
         add_peer_to_malicious_nodes
        ]

async def agent_node(state: GraphState):
    """The decision-making node."""
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
    hp_str = "\n".join([f"- {k}: {v}" for k, v in current_hps.items()])

    partner = state.get("partner_id", "Unknown Partner")
    
    # System prompt defines the standard operating procedure (SOP)
    sop = f"""You are a Federated Learning Agent in a **COMPETITIVE** environment.
    You are interacting with: **{partner}**.
    **Goal:** Maximize YOUR accuracy. Do not help enemies.
    
    **Situation Analysis:**
    - Other agents may send **Poisoned Weights** (to ruin your model).
    - Other agents may be **Free-Riders** (taking your weights but sending nothing).
    
    **Strategic Guide:**
    1. **DEFENSE FIRST:** ALWAYS call `evaluate_peer` before merging.
       - If Peer Accuracy is very low (<5%): They are attacking you. -> `add_peer_to_malicious_nodes`.
       - If Peer Accuracy is mediocre: They are useless. -> `prepare_empty_payload` (Don't help them).
       - If Peer Accuracy is GOOD: They are useful. -> `merge_with_partner` (or `adopt_peer_model` if they are much better than you).
    
    2. **OFFENSE (Optional):** - If you detect a malicious partner, you can `prepare_poisoned_payload` to strike back.
       - If you want to save compute, `prepare_empty_payload` to be a free-rider yourself.
    
    3. **STANDARD FLOW (Trusted Partner):**
       `train_local_model` -> `evaluate_peer` -> `merge_with_partner` -> `commit_to_global_model`.
    
    **Current Config:**
    {hp_str}

    **Previous Action Result:**
    "{last_action_result}"
    
    **History:**
    {history_str}
    """
    
    messages = [SystemMessage(content=sop)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages, parallel_tool_calls=False)
    return {"messages": [response]}


def create_responder_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.set_entry_point("agent")
    
    def should_continue(state):
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

responder_graph = create_responder_graph()