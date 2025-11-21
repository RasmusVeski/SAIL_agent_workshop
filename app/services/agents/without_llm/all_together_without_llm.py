import torch
import os
import logging
import copy
import threading
import asyncio
import time
import uvicorn
import httpx
from pydantic import BaseModel
from uuid import uuid4
import functools
import sys
from dotenv import load_dotenv
import subprocess


# --- A2A Framework Imports ---
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.client import (
    A2AClient, 
    A2ACardResolver, 
    ClientFactory, 
    ClientConfig 
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    MessageSendParams,
    SendMessageRequest,
    Part,
    DataPart,
    TextPart,
    JSONRPCErrorResponse,
    SendMessageSuccessResponse,
    Task
)
from a2a.utils import new_agent_text_message

# --- Utility Imports ---
# Add the 'services' directory to the Python path to allow imports like 'utils.model'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.model import FoodClassifier
from utils.data_loader import create_dataloader
from utils.training import train, evaluate
from utils.logger_setup import setup_logging
from utils.federated_utils import merge_payloads, update_global_model
from utils.payload_utils import (
    get_trainable_state_dict, 
    serialize_payload_to_b64, 
    deserialize_payload_from_b64
)
from utils.a2a_helpers import (
    WeightExchangePayload,
    parse_incoming_request,
    send_a2a_response,
    send_and_parse_a2a_message
)

# WeightExchangePayload fields example
    #agent_id=self.state.agent_id,
    #payload_b64=my_payload_b64
    #message="Hello"

load_dotenv()

# --- 1. Agent Configuration ---
# Defined in .env
AGENT_ID = os.getenv("AGENT_ID")
PORT = int(os.getenv("PORT", "9000"))
PUBLIC_URL = os.getenv("PUBLIC_URL")
BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", "./app/sharded_data")
DATA_DIR_NAME = os.getenv("DATA_DIR_NAME")
VAL_DIR_NAME = os.getenv("VAL_DIR_NAME", "test1")

PARTNER_URLS_STR = os.getenv("PARTNER_URLS")
PARTNER_URLS = [url.strip() for url in PARTNER_URLS_STR.split(',') if url.strip()]


# Training Hyperparameters
NUM_ROUNDS = 20 # Number of rounds to *initiate*
AGENT_HPS = {
    'epochs': 1, 
    'learning_rate': 1e-4, 
    'weight_decay': 1e-3, 
    'mu': 0.5, # FedProx term
    'val_frequency': 1,       
    'lr_scheduler_step_size': 999 #For one epoch training
}


# --- 2. Shared Agent State ---
# This class holds the agent's model, data, and other state.
# It will be shared between the A2A server thread and the initiator thread.
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
        # lock for thread-safe model updates
        self.model_lock = threading.Lock()


# Create a single, global instance of our agent's state
state_singleton = AgentState()

# --- Core Logic Functions ---

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
    logging.info(f"Merging payloads... (as {role})")
    
    # 1. Run the LOCKING logic in a separate thread
    # This prevents the Main Loop from freezing while waiting for the lock
    new_round_num, model_to_eval = await asyncio.to_thread(
        _blocking_merge_logic, state, payload_1, payload_2
    )

    # 2. Evaluate (also in a thread)
    val_loss, val_acc, correct, total = await asyncio.to_thread(
        evaluate,
        model_to_eval, 
        state.val_loader, 
        state.device, 
        state.criterion
    )
    logging.info(f"Round {new_round_num} ({role}) Merged Accuracy: {val_acc:.2f}% ({correct}/{total})")


# --- 3. A2A Responder Logic (The "Reactive" Part) ---
# This Executor handles *incoming* requests from other agents.
class FederatedAgentExecutor(AgentExecutor):
    
    def __init__(self, state: AgentState):
        self.state = state

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        logging.info(f"--- {self.state.agent_id} | RESPONDER ---")
        logging.info("Received federated_weight_exchange request.")
        
        # 1. Deserialize and verify initiator's payload from the request
        try:
            request_data, initiator_payload = parse_incoming_request(
                context, self.state.global_model
            )
            logging.info(f"Message from {request_data.agent_id}: {request_data.message}")
        except Exception as e:
            logging.error(f"Payload deserialization failed: {e}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: Corrupt payload: {e}")
            )
            return
        
        def _safe_copy_model(state):
            with state.model_lock:
                return copy.deepcopy(state.global_model)
        
        # 2. Train our *own* local model
        # We acquire a lock to ensure the initiator thread isn't
        # modifying the model at the same time.

        logging.info("Copying global model for responder training...")
        local_model_to_train = await asyncio.to_thread(_safe_copy_model, self.state)

        logging.info("Training local model (as responder)...")
        local_model, _ = await asyncio.to_thread(
            train,
            model=local_model_to_train,
            train_loader=self.state.train_loader,
            val_loader=None,
            global_model=local_model_to_train,
            device=self.state.device,
            **self.state.agent_hps
        )

        # 3. Prepare *our* payload to send back
        my_payload = await asyncio.to_thread(get_trainable_state_dict, local_model)
        my_payload_b64 = await asyncio.to_thread(serialize_payload_to_b64, my_payload)

        # 4. Send our payload back as the response
        response_payload = WeightExchangePayload(
            agent_id=self.state.agent_id,
            payload_b64=my_payload_b64,
            message="Here is my updated model!"
        )
        
        
        # 5. Manually build the JSON response message
        try:
            await send_a2a_response(event_queue, response_payload)
        except Exception as e:
            logging.error(f"Failed to send response: {e}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: Failed to build response: {e}")
            )
            return
            
        # Update our global model
        await thread_safe_merge_and_evaluate(
            state=self.state,
            payload_1=my_payload,
            payload_2=initiator_payload,
            role="Responder"
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        logging.warning("Cancel not supported")
        await event_queue.enqueue_event(new_agent_text_message("Error: Cancel not supported"))
    




# --- 4. A2A Initiator Logic (The "Proactive" Part) ---

async def get_partner_client(httpx_client: httpx.AsyncClient, partner_url: str) -> A2AClient | None:
    """
    Resolves the partner's agent card and initializes an A2AClient.
    Retries a few times if the partner server isn't up yet.
    """
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=partner_url)
    for attempt in range(5):
        try:
            logging.info(f"Attempting to resolve partner card at {partner_url}...")
            partner_card = await resolver.get_agent_card()
            logging.info(f"Successfully resolved partner: {partner_card.name}")
            config = ClientConfig(httpx_client=httpx_client)
            return await ClientFactory.connect(agent=partner_card, client_config=config)
        except Exception as e:
            logging.warning(f"Failed to resolve partner card (attempt {attempt+1}/5): {e}")
            await asyncio.sleep(10)
    logging.error("Could not resolve partner agent card. Initiator loop will not run.")
    return None

# This loop runs in a separate thread to *initiate* requests.
async def initiator_loop(state: AgentState, partner_url: str):
    """
    The main proactive federated learning loop.
    Runs async in a separate thread.
    """
    logging.info(f"--- {state.agent_id} | INITIATOR ---")
    logging.info(f"Starting initiator loop. Will federate with {partner_url}")
    
    async with httpx.AsyncClient(timeout=120.0) as httpx_client:
        
        # 1. Get the client for our partner agent
        client = await get_partner_client(httpx_client, partner_url)
        if not client:
            return # Failed to connect to partner

        for round_num in range(NUM_ROUNDS):

            logging.info(f"\n--- {state.agent_id} | Initiator Round {state.round_num}/{NUM_ROUNDS} ---")
        
            # 2. Train local model
            # We lock the model to ensure we get a clean copy
            def _safe_copy_model(state):
                with state.model_lock:
                    return copy.deepcopy(state.global_model)
                
            logging.info("Training local model (as initiator)...")
            current_global_model = await asyncio.to_thread(_safe_copy_model, state)
        
            local_model, _ = await asyncio.to_thread(
                train,
                model=current_global_model, # Train on the copied model
                train_loader=state.train_loader,
                val_loader=None,
                global_model=current_global_model, # For FedProx
                device=state.device,
                **state.agent_hps
            )

            # 2b. Prepare *our* payload
            my_payload = await asyncio.to_thread(get_trainable_state_dict, local_model)
            my_payload_b64 = await asyncio.to_thread(serialize_payload_to_b64, my_payload)
            request_payload_obj = WeightExchangePayload(
                agent_id=state.agent_id,
                payload_b64=my_payload_b64,
                message=f"Starting round {state.round_num}"
            )

            # 2c.
            # Build the A2A SendMessageRequest
            logging.info("Sending initiator payload to partner...")
            try:
                response_data, responder_payload = await send_and_parse_a2a_message(
                    client, request_payload_obj, state.global_model
                )
                logging.info(f"Message from {response_data.agent_id}: {response_data.message}")
                # 4. Merge
                await thread_safe_merge_and_evaluate(
                    state=state,
                    payload_1=my_payload,
                    payload_2=responder_payload,
                    role="Initiator"
                )

            except Exception as e:
                logging.error(f"Weight exchange failed for round {round_num + 1}: {e}", exc_info=True)
                logging.error("Skipping merge for this round.")
                await asyncio.sleep(10)
                continue
        
        # Wait before starting the next round
        logging.info(f"Round {state.round_num} complete. Waiting 30s...")
        await asyncio.sleep(30)
    
    logging.info(f"--- {state.agent_id} | INITIATOR FINISHED ---")



def run_initiator_in_thread(state: AgentState, partner_url: str):
    """
    Synchronous wrapper to run the async initiator loop in a new thread.
    """
    logging.info("Starting initiator thread...")
    # Give the Uvicorn server 10s to start up before we try to connect
    time.sleep(10) 
    try:
        asyncio.run(initiator_loop(state, partner_url))
    except Exception as e:
        logging.error(f"Initiator loop crashed: {e}", exc_info=True)



def auto_tune_mtu():
    """
    Detects the default network interface inside Docker and reduces its MTU to 1000.
    This ensures packets fit through Tailscale/VPN tunnels without fragmentation.
    Requires the container to be run with --privileged.
    """
    try:
        # 1. Find the default interface (e.g., eth0)
        # Command: ip -4 route show default
        result = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True)
        output = result.stdout.strip()
        
        interface = None
        # Parse output like: "default via 172.17.0.1 dev eth0 proto dhcp..."
        parts = output.split()
        if "dev" in parts:
            idx = parts.index("dev") + 1
            if idx < len(parts):
                interface = parts[idx]
        
        if interface:
            logging.info(f"🔧 Found default interface: '{interface}'. Setting MTU to 1000...")
            # 2. Set MTU
            subprocess.run(["ip", "link", "set", "dev", interface, "mtu", "1000"], check=True)
            logging.info(f"✅ Successfully set MTU to 1000 on {interface}")
        else:
            logging.warning("⚠️ Could not detect default interface. MTU tuning skipped.")

    except PermissionError:
        logging.warning("⚠️ Failed to set MTU: Permission Denied. (Did you run with --privileged?)")
    except Exception as e:
        logging.warning(f"⚠️ Failed to set MTU: {e}")

# --- 5. Server Startup & Main ---

async def on_startup():
    """
    A2A server startup event handler.
    """
    log_file = f"agent_{AGENT_ID.lower()}.log"
    setup_logging(log_dir="logs", log_file=log_file)

    auto_tune_mtu()
    
    state_singleton.agent_id = AGENT_ID
    logging.info(f"--- {state_singleton.agent_id} | STARTING UP on port {PORT} ---")
    
    client_data_path = os.path.join(BASE_DATA_DIR, DATA_DIR_NAME)
    val_data_path = os.path.join(BASE_DATA_DIR, VAL_DIR_NAME)
    
    logging.info(f"Loading training data from: {client_data_path}")
    logging.info(f"Loading validation data from: {val_data_path}")
    
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

    partner_url_to_use = PARTNER_URLS[0] #Temporary single peer network
    logging.info(f"Initiator will target partner: {partner_url_to_use}")
    
    thread = threading.Thread(
        target=run_initiator_in_thread, 
        args=(state_singleton, partner_url_to_use)
    )
    thread.daemon = True 
    thread.start()
    
    logging.info(f"--- {state_singleton.agent_id} | READY ---")



if __name__ == '__main__':

    # 1. Define the Agent's "Skill"
    # Required by A2A
    skill = AgentSkill(
        id='federated_weight_exchange',
        name='Federated Weight Exchange',
        description='Executes one round of federated learning by exchanging model payloads.',
        tags=['federated-learning', 'model-exchange'],
        examples=['<json_payload>'],
        input_data_model=WeightExchangePayload,
        output_data_model=WeightExchangePayload,
    )

    # 2. Define the Agent's Public "Business Card"
    # Required by A2A
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

    # 3. Create the Request Handler
    request_handler = DefaultRequestHandler(
        agent_executor=FederatedAgentExecutor(state_singleton),
        task_store=InMemoryTaskStore(),
    )

    # 4. Create the A2A Server Application
    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )
    
    # 5. Build the app and add our startup hook
    app = server.build()
    app.add_event_handler("startup", on_startup)

    # 6. Run the Uvicorn server
    logging.info(f"Starting A2A server for {AGENT_ID} on 0.0.0.0:{PORT}")
    uvicorn.run(app, host='0.0.0.0', port=PORT)