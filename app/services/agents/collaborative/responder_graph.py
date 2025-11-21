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
from services.agents.collaborative.state import state_singleton, _blocking_merge_logic
# (We import the singleton because tools need access to the GLOBAL weights)

# --- Utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.training import train, evaluate
from utils.payload_utils import get_trainable_state_dict

# --- 1. The Graph State ---
# This tracks the *conversation* and the *request context*, not the model weights.
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    initiator: str


# --- 2. The Tools ---

@tool
def train_local_model():
    """
    Trains the local model on the agent's private dataset.
    This creates the new weights that will be sent to the partner and merged.
    """
    logging.info("RESPONDER: [Tool] Starting local training...")
    
    # 1. Safe Copy (Thread-safe access to global model)
    def _safe_copy():
        with state_singleton.model_lock:
            return copy.deepcopy(state_singleton.global_model)
    
    # 2. Train
    local_model_to_train = _safe_copy()
    
    trained_model, history = train(
        model=local_model_to_train,
        train_loader=state_singleton.train_loader,
        val_loader=None,
        global_model=local_model_to_train,
        device=state_singleton.device,
        **state_singleton.agent_hps
    )
    
    # 3. Save results to Singleton
    trainable_dict = get_trainable_state_dict(trained_model)
    state_singleton.responder_local_weights = trainable_dict
    
    loss = history[-1]['train_loss'] if history else "N/A"
    return f"Training complete. Final Loss: {loss:.4f}. Weights stored in shared state."


@tool
def merge_and_update():
    """
    Merges the locally trained weights with the partner's incoming weights
    and updates the Global Model.
    """
    logging.info("RESPONDER: [Tool] Merging models...")
    
    if not hasattr(state_singleton, 'incoming_payload'):
        return "Error: No incoming payload found to merge with."

    if hasattr(state_singleton, 'responder_local_weights'):
        local_payload = state_singleton.responder_local_weights
        action_note = "Trained & Merged"
    else:
        logging.warning("RESPONDER: No training detected. Merging using CURRENT global weights.")
        with state_singleton.model_lock:
            local_payload = get_trainable_state_dict(state_singleton.global_model)
        action_note = "Skipped Training & Merged"

    # 1. Merge (Thread Safe)
    new_round, model_for_eval = _blocking_merge_logic(
        state_singleton,
        local_payload,
        state_singleton.incoming_payload
    )
    
    # 2. Evaluate (Can be done outside the lock now that we have a copy)
    val_loss, val_acc, correct, total = evaluate(
        model_for_eval,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion
    )

    result_str = f"Acc: {val_acc:.2f}% ({correct}/{total})"
    
    history_entry = {
        "round": new_round,
        "role": "RESPONDER",
        "partner": "Partner", # Todo: Pass partner ID into this tool
        "action": action_note,
        "result": result_str
    }
    state_singleton.log_history(history_entry)
    
    return f"Global Model updated to Round {new_round}. {result_str}"

@tool
def evaluate_model():
    """
    Runs evaluation on the current Global Model using the validation dataset.
    Returns the accuracy and loss. Use this to check if your training helped.
    """
    logging.info("RESPONDER: [Tool] Evaluating model...")
    
    # Safe copy for eval (or just eval directly since we aren't mutating)
    with state_singleton.model_lock:
        model_to_eval = copy.deepcopy(state_singleton.global_model)
        
    val_loss, val_acc, correct, total = evaluate(
        model_to_eval,
        state_singleton.val_loader,
        state_singleton.device,
        state_singleton.criterion
    )
    result_str = f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%"
    
    # Log results to general memory
    history_entry = {
        "round": state_singleton.round_num,
        "role": "RESPONDER",
        "partner": "Self (Eval)",
        "action": "Manual Evaluation",
        "result": result_str
    }
    state_singleton.log_history(history_entry)
    
    return f"Validation Results - {result_str}"


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

tools = [train_local_model, merge_and_update, evaluate_model, update_training_parameters]

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
    
    # System prompt defines the standard operating procedure (SOP)
    sop = f"""You are an autonomous Federated Learning Agent (Responder).
    **Your Learning History:**
    {history_str}

    **Current Training Hyperparameters:**
    {hp_str}
    
    **Goal:** Collaborate to improve accuracy. 
    - If performance is dropping, consider changing hyperparameters.
    - Always Train and Merge unless you have a specific reason not to.
    
    Your final text response will be sent back to the partner.
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