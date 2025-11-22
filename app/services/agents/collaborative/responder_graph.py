import logging
import json
import os
import sys
import asyncio
import copy
from typing import Annotated, List, TypedDict, Any
from uuid import uuid4
import operator

# --- LangChain / LangGraph Imports ---
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# --- Local Imports ---
from services.agents.collaborative.state import state_singleton, _blocking_commit_logic
# (We import the singleton because tools need access to the GLOBAL weights)

# --- Utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.training import train, evaluate
from utils.federated_utils import merge_payloads
from utils.payload_utils import get_trainable_state_dict, deserialize_payload_from_b64

# --- 1. The Graph State ---
# This tracks the *conversation* and the *request context*, not the model weights.
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    partner_id: str


# --- 2. Workspace management ---
def _get_or_create_working_copy():
    """
    Ensures 'responder_working_weights' exists.
    If not, creates it from the current global model.
    """
    if state_singleton.responder_working_weights is None:
        logging.info("RESPONDER: Creating new working copy from Global Model.")
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
    logging.info("RESPONDER: [Tool] Starting local training...")
    
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
        **state_singleton.agent_hps
    )
    
    # 4. Update Workspace
    new_weights = get_trainable_state_dict(trained_model)
    _update_working_copy(new_weights)
    
    loss = history[-1]['train_loss'] if history else "N/A"
    return f"Training complete. Final Loss: {loss:.4f}. Local draft updated."


@tool
def merge_with_partner(partner_id: str, merge_alpha: float = 0.5):
    """
    Merges the Partner's incoming weights into your Local Draft.
    Does NOT update the Global Model.
    0 <= merge_alpha <= 1, the bigger the more of local model retained
    """
    logging.info(f"RESPONDER: [Tool] Merging with {partner_id}...")
    
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
    
    return "Merge complete. Local draft now contains combined weights. Ready to evaluate or commit."

@tool
def evaluate_model():
    """
    Evaluates the current model state. 
    If you have a Local Draft, it evaluates that.
    If you don't, it evaluates the Global Model.
    """
    logging.info("RESPONDER: [Tool] Evaluating model...")
    
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
    val_loss, val_acc, correct, total = evaluate(
        model_instance,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion
    )
    
    return f"[{target_name}] Validation Results - Accuracy: {val_acc:.2f}% ({correct}/{total})"


@tool
def commit_to_global_model(partner_id: str):
    """
    Commits the Local Draft to the Global Model.
    This is the FINAL step. It mixes the draft with the current Global Model (Safety Anchor) and saves it.
    """
    logging.info("RESPONDER: [Tool] Committing to Global...")

    if state_singleton.responder_working_weights is None:
        return "Error: No Local Draft exists. Train or Merge first."

    # 1. Blocking Commit (Thread Safe)
    # This does the 80/20 merge with the live global model
    new_round, model_for_eval = _blocking_commit_logic(
        state_singleton,
        state_singleton.responder_working_weights
    )
    
    # 2. Quick Eval of the final result
    val_loss, val_acc, correct, total = evaluate(
        model_for_eval,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion
    )
    
    result_str = f"Acc: {val_acc:.2f}%"

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
    
    return f"Success! Global Model updated to Round {new_round}. Final Acc: {val_acc:.2f}%"


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
    logging.info(f"RESPONDER: [Tool] Updating HPs: lr={learning_rate}, epochs={epochs}, mu={mu}")
    
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

tools = [train_local_model, merge_with_partner, evaluate_model, commit_to_global_model, update_training_parameters]

async def agent_node(state: GraphState):
    """The decision-making node."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "openai/gpt-oss-120b"),
        base_url=os.getenv("OPENAI_API_BASE", "https://inference.rcp.epfl.ch/v1"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    llm_with_tools = llm.bind_tools(tools)

    history_str = state_singleton.get_formatted_history()
    current_hps = state_singleton.agent_hps
    hp_str = "\n".join([f"- {k}: {v}" for k, v in current_hps.items()])

    partner = state.get("partner_id", "Unknown Partner")
    
    # System prompt defines the standard operating procedure (SOP)
    sop = f"""You are an autonomous Federated Learning Agent (Responder).
    You are currently communicating with: **{partner}**.
    **Your Learning History:**
    {history_str}

    **Current Training Hyperparameters:**
    {hp_str}
    
    **Goal:** Collaborate to improve accuracy.
    **Standard Workflow (You may deviate if you have a reason):**
    1. `train_local_model` (Creates a local draft from your data).
    2. `merge_with_partner` (Combines your draft with the partner's data).
    3. `evaluate_model` (Checks if the draft is actually good).
    4. `commit_to_global_model` (If good, saves it to the permanent model. Pass '{partner}' as partner_id).
    5. Respond to the partner with a text message.

    If the model is getting worse, consider updating hyperparameters or not committing.

    Generally, to achieve the best results, you should train one epoch and then merge and commit.
    """
    
    messages = [SystemMessage(content=sop)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
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