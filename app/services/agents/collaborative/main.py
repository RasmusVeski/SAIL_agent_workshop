import logging
import threading
import uvicorn
import os
import sys
import torch
import subprocess
from dotenv import load_dotenv

# --- A2A Imports ---
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# --- Local Imports ---
from services.agents.collaborative.state import state_singleton
from services.agents.collaborative.responder import CollaborativeResponderExecutor
from services.agents.collaborative.initiator import run_initiator_in_thread

# --- Utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.model import FoodClassifier
from utils.data_loader import create_dataloader
from utils.training import evaluate
from utils.logger_colored import setup_logging
from utils.a2a_helpers import WeightExchangePayload

load_dotenv()

# --- Configuration ---
AGENT_ID = os.getenv("AGENT_ID")
PORT = int(os.getenv("PORT", "9000"))
PUBLIC_URL = os.getenv("PUBLIC_URL")
BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", "./app/sharded_data")
DATA_DIR_NAME = os.getenv("DATA_DIR_NAME")
VAL_DIR_NAME = os.getenv("VAL_DIR_NAME", "test1")

PARTNER_URLS_STR = os.getenv("PARTNER_URLS")
PARTNER_URLS = [url.strip() for url in PARTNER_URLS_STR.split(',') if url.strip()]

def auto_tune_mtu():
    """
    Detects default network interface inside Docker 
    and reduces MTU to 1000. Requires --privileged.
    """
    try:
        result = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True)
        output = result.stdout.strip()
        interface = None
        parts = output.split()
        if "dev" in parts:
            idx = parts.index("dev") + 1
            if idx < len(parts):
                interface = parts[idx]
        
        if interface:
            logging.info(f"🔧 Found default interface: '{interface}'. Setting MTU to 1000...")
            subprocess.run(["ip", "link", "set", "dev", interface, "mtu", "1000"], check=True)
            logging.info(f"✅ Successfully set MTU to 1000 on {interface}")
        else:
            logging.warning("⚠️ Could not detect default interface. MTU tuning skipped.")
    except PermissionError:
        logging.warning("⚠️ Failed to set MTU: Permission Denied. (Did you run with --privileged?)")
    except Exception as e:
        logging.warning(f"⚠️ Failed to set MTU: {e}")

async def on_startup():
    setup_logging(log_dir="logs", agent_id=AGENT_ID)
    
    # Fix Network
    auto_tune_mtu()
    
    state_singleton.agent_id = AGENT_ID
    logging.info(f"--- {state_singleton.agent_id} | STARTING UP on port {PORT} ---")
    
    client_data_path = os.path.join(BASE_DATA_DIR, DATA_DIR_NAME)
    val_data_path = os.path.join(BASE_DATA_DIR, VAL_DIR_NAME)
    
    logging.info(f"Loading training data from: {client_data_path}")
    state_singleton.train_loader = create_dataloader(client_data_path, 32, shuffle=True, num_workers=0)
    state_singleton.val_loader = create_dataloader(val_data_path, 32, shuffle=False, num_workers=0)
    
    if not state_singleton.train_loader or not state_singleton.val_loader:
        logging.error("Failed to load data. Server cannot function.")
        return

    state_singleton.global_model = FoodClassifier()
    state_singleton.criterion = torch.nn.CrossEntropyLoss()
    
    val_loss, val_acc, _, _ = evaluate(
        state_singleton.global_model, state_singleton.val_loader, state_singleton.device, state_singleton.criterion
    )
    logging.info(f"Round 0 Initial Accuracy: {val_acc:.2f}%")

    if PARTNER_URLS:
        logging.info(f"Initiator configured with {len(PARTNER_URLS)} potential partners.")
        
        thread = threading.Thread(
            target=run_initiator_in_thread, 
            args=(state_singleton, PARTNER_URLS) # Pass the whole list
        )
        thread.daemon = True 
        thread.start()
    
    logging.info(f"--- {state_singleton.agent_id} | READY ---")

if __name__ == '__main__':
    skill = AgentSkill(
        id='federated_weight_exchange',
        name='Federated Weight Exchange',
        description='Executes one round of federated learning.',
        tags=['federated-learning', 'model-exchange'],
        examples=['<json_payload>'],
        input_data_model=WeightExchangePayload,
        output_data_model=WeightExchangePayload,
    )

    public_agent_card = AgentCard(
        name=f'Federated Learning Agent ({AGENT_ID})',
        description='An agent that participates in federated learning.',
        url=PUBLIC_URL,
        version='1.0.0',
        default_input_modes=['data'],
        default_output_modes=['data'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supports_authenticated_extended_card=False,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=CollaborativeResponderExecutor(state_singleton),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )
    
    app = server.build()
    app.add_event_handler("startup", on_startup)
    
    logging.info(f"Starting A2A server for {AGENT_ID} on 0.0.0.0:{PORT}")
    uvicorn.run(app, host='0.0.0.0', port=PORT)