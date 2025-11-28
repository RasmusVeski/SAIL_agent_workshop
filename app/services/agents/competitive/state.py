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
    'learning_rate': 1e-4, 
    'weight_decay': 1e-3, # L2, penalty to big weights
    'mu': 0.5,
    'val_frequency': 1,       
    'lr_scheduler_step_size': 999 # Hack for keeping same learning rate since train for few epochs
}

class AgentState:
    def __init__(self):
        self.agent_id = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = None # protected by `model_lock`
        self.train_loader = None
        self.val_loader = None
        self.criterion = None # CrossEntropyLoss
        self.round_num = 0 # Counter for the current global training round/update.
        self.agent_hps = AGENT_HPS # The current set of hyperparameters used for the local training loop.

        # --- Network State ---
        self.available_partners = [] # List of partner base URLs (strings) that this agent can initiate contact with.
        self.shared_httpx_client = None # The shared HTTP client object used by both the Initiator and Responder to make requests.

        # --- Shared Episodic Memory ---
        # Stores dicts: {'round': 1, 'role': 'RESPONDER', 'partner': 'Agent_B', 'result': 'Acc: 10%', 'action': 'Merged'}
        self.history = [] # A chronological list of past actions, partners, and results. Used to inform the LLM's next strategic decision.

        # --- Thread-Specific Scratchpads ---
        # These are temporary holders for model weights/payloads during a single P2P exchange.
        self.responder_working_weights = None # The model weights *after* the Responder trains locally (the draft update).
        self.responder_incoming_payload = None # The model weights received *from* an Initiator partner.
        self.responder_partner_id = None # The ID of the partner the responder is conversation with
        self.responder_outbound_payload = None # Stores the specific payload the Responder WANTS to send, can be poisoned

        # Initiator Scratchpad
        self.initiator_working_weights = None # The model weights *after* the Initiator trains locally (the draft update).
        self.initiator_incoming_payload = None # The model weights received *from* the Responder partner.
        self.active_client = None # The A2AClient object for the Initiator thread to use
        self.current_partner_id = None # The ID of the partner being communicated with during the current active session.

        # --- Lock ---
        self.model_lock = threading.Lock() # standard thread lock to protect the self.global_model and self.round_num.
        self.responder_lock = asyncio.Lock() # An async lock to prevent multiple incoming responder requests from being processed simultaneously.

        # --- Competetive ---
        self.malicious_nodes = set() # Agent IDs of nodes you do not trust


    def log_history(self, entry: dict):
        """Thread-safe append to history."""
        with self.model_lock:
            self.history.append(entry)
            # Keep only last 50 entries to prevent infinite growth
            if len(self.history) > 50:
                self.history.pop(0)

    def get_formatted_history(self, limit=10) -> str:
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

def blocking_commit(state: AgentState, draft_payload: dict, alpha: float = 0.2):
    """
    Commits the 'Draft' (working weights) to the Global Model.
    
    Args:
        alpha: The weight of the CURRENT Global Model in the merge.
               0.0 = Completely Overwrite Global (Commit without merging).
               0.2 = Keep 20% Old, 80% New.
               0.5 = Balanced Merge.
    """
    with state.model_lock:
        # 1. Get Current Global
        current_global_payload = get_trainable_state_dict(state.global_model)
        
        # 2. Merge Global (Anchor) with Draft (New Knowledge)
        # merge_payloads(A, B, alpha) -> alpha*A + (1-alpha)*B
        final_merge = merge_payloads(current_global_payload, draft_payload, alpha=alpha)

        # 3. Update
        update_global_model(state.global_model, final_merge)
        state.round_num += 1 
        new_round_num = state.round_num
        
        # 4. Copy for eval
        model_to_eval = copy.deepcopy(state.global_model)
        
    return new_round_num, model_to_eval