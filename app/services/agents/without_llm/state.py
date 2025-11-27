import torch
import threading
import logging
import copy
import asyncio
import os
import sys

# Utils
from services.utils.federated_utils import merge_payloads, update_global_model
from services.utils.payload_utils import get_trainable_state_dict
from services.utils.training import evaluate

# --- Configuration Constants ---
NUM_ROUNDS = 20
AGENT_HPS = {
    'epochs': 1, 
    'learning_rate': 0.002, 
    'weight_decay': 1e-4, 
    'mu': 0.2,
    'val_frequency': 1,       
    'lr_scheduler_step_size': 999 
}

class AgentState:
    def __init__(self):
        self.agent_id = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.round_num = 0
        self.agent_hps = AGENT_HPS
        self.model_lock = threading.Lock()

# The Global Singleton
state_singleton = AgentState()

# --- Shared Logic (Used by both Initiator and Responder) ---

def _blocking_merge_logic(state: AgentState, payload_1: dict, payload_2: dict):
    """
    Synchronous function that holds the lock and updates the model.
    Safe to run in a thread.
    """
    with state.model_lock:
        # 1. Merge
        merged_payload = merge_payloads(payload_1, payload_2, alpha=0.5)
        # 2. Update
        update_global_model(state.global_model, merged_payload)
        # 3. Increment round
        state.round_num += 1 
        new_round_num = state.round_num
        # 4. Deepcopy for eval
        model_to_eval = copy.deepcopy(state.global_model)
        
    return new_round_num, model_to_eval

async def thread_safe_merge_and_evaluate(state: AgentState, payload_1: dict, payload_2: dict, role: str):
    logging.info(f"{role}: Merging payloads...")
    
    # 1. Run the LOCKING logic in a separate thread
    # This prevents the Main Loop from freezing while waiting for the lock
    new_round_num, model_to_eval = await asyncio.to_thread(
        _blocking_merge_logic, state, payload_1, payload_2
    )

    # 2. Evaluate (also in a thread)
    val_loss, val_acc, correct, total, classes_learned = await asyncio.to_thread(
        evaluate,
        model_to_eval, 
        state.val_loader, 
        state.device, 
        state.criterion
    )
    logging.info(f"{role}: Round {new_round_num} ({role}) Merged Accuracy: {val_acc:.2f}% ({correct}/{total}) | Classes: {classes_learned}/{40}")