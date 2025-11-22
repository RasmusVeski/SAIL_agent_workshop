import torch
import threading
import logging
import copy
import asyncio
import os
import sys

# Adjust path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.federated_utils import merge_payloads, update_global_model
from utils.payload_utils import get_trainable_state_dict
from utils.training import evaluate

# --- Configuration Constants ---
NUM_ROUNDS = 20
AGENT_HPS = {
    'epochs': 1, 
    'learning_rate': 1e-4, 
    'weight_decay': 1e-3, 
    'mu': 0.5,
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

        # --- Network State ---
        self.available_partners = [] # List of URLs strings
        self.shared_httpx_client = None # Assigned by initiator loop

        # --- Shared Episodic Memory ---
        # Stores dicts: {'round': 1, 'role': 'RESPONDER', 'partner': 'Agent_B', 'result': 'Acc: 10%', 'action': 'Merged'}
        self.history = []

        # --- Thread-Specific Scratchpads ---
        # Responder Scratchpad
        self.responder_local_weights = None
        self.responder_incoming_payload = None # Responder's incoming data

        # Initiator Scratchpad
        self.initiator_local_weights = None
        self.initiator_incoming_payload = None
        self.active_client = None # The A2AClient object for the Initiator thread to use

        # --- Lock ---
        self.model_lock = threading.Lock()


    def log_history(self, entry: dict):
        """Thread-safe append to history."""
        with self.model_lock:
            self.history.append(entry)
            # Keep only last 50 entries to prevent infinite growth
            if len(self.history) > 50:
                self.history.pop(0)

    def get_formatted_history(self, limit=5) -> str:
        """Returns a formatted string of recent history for the LLM."""
        # We don't need a lock just to read/copy the list for a prompt
        recent = self.history[-limit:]
        if not recent:
            return "No history yet."
        
        lines = []
        for entry in recent:
            line = (f"* [Round {entry.get('round')} | {entry.get('role')}] "
                    f"Partner: {entry.get('partner')} | "
                    f"Action: {entry.get('action')} | "
                    f"Result: {entry.get('result')}")
            lines.append(line)
        return "\n".join(lines)

# The Global Singleton
state_singleton = AgentState()

# --- Shared Logic (Used by both Initiator and Responder) ---

def _blocking_commit_logic(state: AgentState, draft_payload: dict):
    """
    Commits the 'Draft' (working weights) to the Global Model using a weighted merge.
    This corresponds to the 'Safety Anchor' logic.
    """
    with state.model_lock:
        # 1. Get Current Global
        current_global_payload = get_trainable_state_dict(state.global_model)
        
        # 2. Merge Global (Anchor) with Draft (New Knowledge)
        # Alpha 0.2 = Keep 20% Old Global, Adopt 80% New Draft
        final_merge = merge_payloads(current_global_payload, draft_payload, alpha=0.3)

        # 3. Update
        update_global_model(state.global_model, final_merge)
        state.round_num += 1 
        new_round_num = state.round_num
        
        # 4. Copy for eval
        model_to_eval = copy.deepcopy(state.global_model)
        
    return new_round_num, model_to_eval




# def _blocking_merge_logic(state: AgentState, payload_1: dict, payload_2: dict):
#     """
#     Synchronous function that holds the lock and updates the model.
#     Safe to run in a thread.
#     """
#     with state.model_lock:
#         # 1. Merge the two NEW payloads (The consensus of this specific round)
#         # We weight them 50/50 between the two partners
#         exchange_merge = merge_payloads(payload_1, payload_2, alpha=0.5)
        
#         # 2. Get the CURRENT global model state
#         current_global_payload = get_trainable_state_dict(state.global_model)

#         # 3. Merge the Exchange Result with the Current Global Model
#         # 20% Current Global (Anchor) + 80% New Exchange (Progress)
#         # This is for not completely overwriting results from the other thread, but only take 20%
#         final_merge = merge_payloads(current_global_payload, exchange_merge, alpha=0.2)

#         # 4. Update global model
#         update_global_model(state.global_model, final_merge)
        
#         # 5. Increment round
#         state.round_num += 1 
#         new_round_num = state.round_num
        
#         # 6. Deepcopy for eval (so we can evaluate without holding the lock)
#         model_to_eval = copy.deepcopy(state.global_model)
        
#     return new_round_num, model_to_eval