import logging
import os
import sys
import copy
import random
import asyncio
from typing import Annotated, List, TypedDict
import operator
from uuid import uuid4

# --- LangChain / LangGraph ---
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- A2A ---
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig

# --- Local Imports ---
from services.agents.collaborative.state import state_singleton, _blocking_commit_logic

# --- Utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.training import train, evaluate
from utils.federated_utils import merge_payloads
from utils.payload_utils import get_trainable_state_dict, serialize_payload_to_b64, deserialize_payload_from_b64
from utils.a2a_helpers import WeightExchangePayload, send_and_parse_a2a_message

# --- 1. The Graph State ---
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

# --- 2. Helper: Workspace Management ---
def _get_or_create_working_copy():
    """
    Ensures 'initiator_local_weights' exists.
    If not, creates it from the current global model.
    """
    if state_singleton.initiator_local_weights is None:
        logging.info("INITIATOR: Creating new working copy from Global Model.")
        with state_singleton.model_lock:
            state_singleton.initiator_local_weights = get_trainable_state_dict(state_singleton.global_model)
    return state_singleton.initiator_local_weights

def _update_working_copy(new_weights: dict):
    state_singleton.initiator_local_weights = new_weights

# --- 3. The Tools ---

@tool
async def select_partner_agent():
    """
    Selects a partner agent from the available list and establishes a connection.
    This MUST be called before exchanging weights.
    Choosing a new partner clears any old incoming weights you might have held.
    """
    logging.info("INITIATOR: [Tool] Selecting partner...")
    
    partners = state_singleton.available_partners
    if not partners:
        return "Error: No partners available in configuration."
    
    # Random selection for now (Agent can't choose specific yet)
    target_url = random.choice(partners)
    
    if not state_singleton.shared_httpx_client:
        return "Error: Network client not initialized. System error."

    # Reset previous exchange state
    state_singleton.initiator_incoming_payload = None
    state_singleton.active_client = None
    
    # Establish Connection
    resolver = A2ACardResolver(state_singleton.shared_httpx_client, base_url=target_url)
    try:
        logging.info(f"INITIATOR: Resolving card for {target_url}...")
        card = await resolver.get_agent_card()
        config = ClientConfig(state_singleton.shared_httpx_client)
        client = await ClientFactory.connect(agent=card, client_config=config)
        
        state_singleton.active_client = client
        return f"Successfully connected to partner: {card.name} at {target_url}. Previous incoming weights cleared."
        
    except Exception as e:
        return f"Failed to connect to {target_url}: {e}"

@tool
def train_local_model():
    """
    Trains the local model on the agent's private dataset.
    Stores the result in the local workspace for the exchange step.
    """
    logging.info("INITIATOR: [Tool] Starting local training...")
    
    # 1. Get Weights (Lazy Load)
    weights_to_train = _get_or_create_working_copy()
    
    # 2. Load into temp model
    def _load_temp_model():
        with state_singleton.model_lock:
            temp_model = copy.deepcopy(state_singleton.global_model)
        curr_state = temp_model.state_dict()
        curr_state.update(weights_to_train)
        temp_model.load_state_dict(curr_state)
        return temp_model

    model_instance = _load_temp_model()
    
    # 3. Train
    import torch
    torch.set_num_threads(2)
    
    trained_model, history = train(
        model=model_instance,
        train_loader=state_singleton.train_loader,
        val_loader=None,
        global_model=model_instance,
        device=state_singleton.device,
        **state_singleton.agent_hps
    )
    
    # 4. Update Workspace
    new_weights = get_trainable_state_dict(trained_model)
    _update_working_copy(new_weights)
    
    loss = history[-1]['train_loss'] if history else "N/A"
    return f"Training complete. Final Loss: {loss:.4f}. Local draft updated."

@tool
async def exchange_weights_with_partner(message_to_partner: str = "Starting exchange"):
    """
    Sends the locally trained weights (or current global weights if untrained) 
    to the active partner and waits for their response.
    """
    logging.info("INITIATOR: [Tool] Exchanging weights...")
    
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
            client, request_payload_obj, state_singleton.global_model
        )
        
        # 4. Stash result
        state_singleton.initiator_incoming_payload = responder_payload
        
        return f"Exchange successful! Partner ({response_data.agent_id}) sent new weights. Message: '{response_data.message}'"
        
    except Exception as e:
        logging.error(f"Tool Exchange failed: {e}")
        return f"Error during exchange: {e}"

@tool
def merge_with_incoming():
    """
    Merges the Partner's incoming weights into your Local Draft.
    Does NOT update the Global Model yet.
    """
    logging.info("INITIATOR: [Tool] Merging incoming weights...")
    
    if not state_singleton.initiator_incoming_payload:
        return "Error: No incoming payload found. Run `exchange_weights_with_partner` first."

    # 1. Get Weights
    local_weights = _get_or_create_working_copy()
    partner_weights = state_singleton.initiator_incoming_payload

    # 2. Merge (50/50)
    merged_weights = merge_payloads(local_weights, partner_weights, alpha=0.5)
    
    # 3. Update Workspace
    _update_working_copy(merged_weights)
    
    return "Merge complete. Local draft now contains combined weights. Ready to evaluate or commit."


@tool
def commit_to_global_model():
    """
    Commits the Local Draft to the Global Model.
    This is the FINAL step. It mixes the draft with the current Global Model (Safety Anchor) and saves it.
    """
    logging.info("INITIATOR: [Tool] Committing to Global...")

    if state_singleton.initiator_working_weights is None:
        return "Error: No Local Draft exists. Train or Merge first."

    # 1. Blocking Commit (Thread Safe)
    new_round, model_for_eval = _blocking_commit_logic(
        state_singleton,
        state_singleton.initiator_working_weights
    )
    
    # 2. Quick Eval
    val_loss, val_acc, correct, total = evaluate(
        model_for_eval,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion
    )
    result_str = f"Acc: {val_acc:.2f}%"

    # 3. Log History
    partner_name = "Unknown"
    # Try to find partner name from active client if it exists
    if state_singleton.active_client:
        partner_name = "Partner" # A2AClient structure makes getting name tricky here without keeping ref
        
    history_entry = {
        "round": new_round,
        "role": "INITIATOR",
        "partner": partner_name, 
        "action": "Committed Update",
        "result": result_str
    }
    state_singleton.log_history(history_entry)
    
    # 4. Clear Workspace
    state_singleton.initiator_working_weights = None
    state_singleton.initiator_incoming_payload = None
    
    return f"Success! Global Model updated to Round {new_round}. Final Acc: {val_acc:.2f}%"

@tool
def evaluate_model():
    """
    Evaluates the current model state.
    If you have a Local Draft, it evaluates that.
    If you don't, it evaluates the Global Model.
    """
    logging.info("INITIATOR: [Tool] Evaluating model...")
    
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
    
    val_loss, val_acc, correct, total = evaluate(
        model_instance,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion
    )
    return f"[{target_name}] Validation Results - Accuracy: {val_acc:.2f}%"

@tool
def update_training_parameters(learning_rate: float = None, epochs: int = None, mu: float = None):
    """Updates hyperparameters."""
    logging.info(f"INITIATOR: [Tool] Updating HPs: lr={learning_rate}, epochs={epochs}, mu={mu}")
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


# --- 3. Graph Construction ---

tools = [
    select_partner_agent, 
    train_local_model, 
    exchange_weights_with_partner, 
    merge_with_incoming, 
    commit_to_global_model, 
    update_training_parameters, 
    evaluate_model
]

async def agent_node(state: GraphState):
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    llm_with_tools = llm.bind_tools(tools)
    
    history_str = state_singleton.get_formatted_history()
    current_hps = state_singleton.agent_hps
    hp_str = ", ".join([f"{k}={v}" for k, v in current_hps.items()])
    
    # Logic to determine if we have partner weights
    has_incoming = "YES" if state_singleton.initiator_incoming_payload else "NO"
    has_active_partner = "YES" if state_singleton.active_client else "NO"
    
    sop = f"""You are an autonomous Federated Learning Agent (Initiator Role).
    You have decided to wake up and consider a training round.
    
    **State Context:**
    - Connected to a Partner: {has_active_partner}
    - Holding Incoming Weights: {has_incoming}
    
    **Your Learning History:**
    {history_str}
    
    **Current Hyperparameters:** {hp_str}
    
    **Goal:** Improve the Global Model.
    
    **Standard Workflow:**
    1. `select_partner_agent` (If not connected or you want to switch).
    2. `train_local_model` (Improve your own knowledge).
    3. `exchange_weights_with_partner` (Send yours, get theirs).
    4. `merge_with_incoming` (Combine them).
    5. `commit_to_global_model` (Save to permanent memory).
    
    You are free to experiment! Change HPs, skip training if performance is good, or evaluate before committing.
    """
    
    messages = [SystemMessage(content=sop)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
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