import torch
import os
import logging
import copy
import uvicorn
import click
import asyncio
import httpx
import random
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Our Federated Learning Utils
from services.utils.model import FoodClassifier
from services.utils.data_loader import create_dataloader
from services.utils.training import train, evaluate
from services.utils.logger_setup import setup_logging
from services.utils.federated_utils import merge_payloads, update_global_model
from services.utils.payload_utils import (
    get_trainable_state_dict, 
    serialize_payload_to_b64, 
    deserialize_payload_from_b64
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Agent State ---
# This class holds our *shared* state, including the model and the lock
class AgentState:
    def __init__(self, agent_id, data_client, peer_url):
        self.agent_id = agent_id
        self.peer_url = peer_url # URL of the agent we will talk to
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.round_num = 0
        self.agent_hps = {
            'epochs': 1, 'learning_rate': 1e-4, 'weight_decay': 1e-3, 
            'mu': 0.5, 'val_frequency': 1, 'lr_scheduler_step_size': 999
        }
        # The critical lock to prevent "stepping on toes"
        self.model_lock = asyncio.Lock()
        
    def load_data_and_model(self, data_client_id, val_client_id):
        """Loads data and initializes the model."""
        logger.info(f"--- {self.agent_id} | Initializing... ---")
        BASE_DIR = "./sharded_data"
        CLIENT_DATA_DIR = os.path.join(BASE_DIR, data_client_id)
        VAL_DIR = os.path.join(BASE_DIR, val_client_id)

        self.train_loader = create_dataloader(CLIENT_DATA_DIR, 32, shuffle=True, num_workers=0)
        self.val_loader = create_dataloader(VAL_DIR, 32, shuffle=False, num_workers=0)
        
        if not self.train_loader or not self.val_loader:
            logger.critical("Failed to load data. Server cannot function.")
            raise RuntimeError("Failed to load data.")

        self.global_model = FoodClassifier()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        val_loss, val_acc, _, _ = evaluate(
            self.global_model, self.val_loader, self.device, self.criterion
        )
        logger.info(f"Round 0 Initial Accuracy: {val_acc:.2f}%")
        logger.info(f"--- {self.agent_id} | READY ---")

# --- 2. FastAPI (Responder Thread) Setup ---
app = FastAPI(title="Federated Learning Agent")

# Pydantic models for request/response
class WeightExchangeRequest(BaseModel):
    agent_id: str
    payload_b64: str

class WeightExchangeResponse(BaseModel):
    agent_id: str
    payload_b64: str

# We need to pass our 'state' to the endpoints.
# A simple (but effective) way is to attach it to the app object.
@app.on_event("startup")
async def startup_event():
    # This is just a placeholder; the real state is created in main()
    # and attached *before* uvicorn.run()
    pass

@app.post("/exchange_weights", response_model=WeightExchangeResponse)
async def exchange_weights(request: WeightExchangeRequest):
    """
    This is the "Responder" logic.
    """
    state: AgentState = app.state.agent_state # Get the shared state
    
    # Try to get the lock. If it's busy (our initiator is working),
    # tell the other agent to try again later.
    if state.model_lock.locked():
        logger.warning(f"SKIPPING exchange with {request.agent_id}, lock is busy.")
        raise HTTPException(status_code=503, detail="Agent is busy, try again later")

    async with state.model_lock:
        state.round_num += 1
        logger.info(f"\n--- {state.agent_id} (Responder) | Round {state.round_num} ---")
        
        # 1. Deserialize initiator's payload
        logger.info(f"Received payload from {request.agent_id}")
        initiator_payload = await asyncio.to_thread(
            deserialize_payload_from_b64,
            request.payload_b64, state.global_model
        )
        if not initiator_payload:
            raise HTTPException(status_code=400, detail="Corrupt payload")

        # 2. Train *our* local model
        logger.info("Training local model...")
        local_model, _ = await asyncio.to_thread(
            train,
            model=copy.deepcopy(state.global_model),
            train_loader=state.train_loader, val_loader=None,
            global_model=state.global_model, device=state.device,
            **state.agent_hps
        )

        # 3. Prepare *our* payload
        logger.info("Preparing responder payload...")
        my_payload = await asyncio.to_thread(get_trainable_state_dict, local_model)
        my_payload_b64 = await asyncio.to_thread(serialize_payload_to_b64, my_payload)
        
        # 4. Merge
        logger.info("Merging payloads...")
        merged_payload = await asyncio.to_thread(
            merge_payloads, my_payload, initiator_payload, 0.5
        )
        
        # 5. Update Global Model
        await asyncio.to_thread(
            update_global_model, state.global_model, merged_payload
        )
        
        # 6. Evaluate
        val_loss, val_acc, _, _ = await asyncio.to_thread(
            evaluate, state.global_model, state.val_loader, state.device, state.criterion
        )
        logger.info(f"Round {state.round_num} Merged Accuracy: {val_acc:.2f}%")
        
        # 7. Send our payload back
        logger.info("Sending responder payload in response.")
        return WeightExchangeResponse(
            agent_id=state.agent_id,
            payload_b64=my_payload_b64
        )

# --- 3. Initiator Thread ---
async def initiator_loop(state: AgentState):
    """
    This is the "Active Talker" logic.
    """
    # Wait a bit for the server to start
    await asyncio.sleep(10) 
    logger.info("--- Initiator loop started ---")
    
    async with httpx.AsyncClient(base_url=state.peer_url, timeout=120.0) as client:
        while True:
            # Wait for a random time before initiating
            await asyncio.sleep(random.randint(15, 30))
            
            # Try to get the lock. If we can't, it means our server
            # is busy, so we skip this round and wait.
            if state.model_lock.locked():
                logger.info(f"--- {state.agent_id} (Initiator) | Skipping round, lock busy. ---")
                continue

            async with state.model_lock:
                state.round_num += 1
                logger.info(f"\n--- {state.agent_id} (Initiator) | Round {state.round_num} ---")
                
                # 1. Train local model
                logger.info("Training local model...")
                local_model, _ = await asyncio.to_thread(
                    train,
                    model=copy.deepcopy(state.global_model),
                    train_loader=state.train_loader, val_loader=None,
                    global_model=state.global_model, device=state.device,
                    **state.agent_hps
                )

                # 2. Prepare *our* payload
                my_payload = await asyncio.to_thread(get_trainable_state_dict, local_model)
                my_payload_b64 = await asyncio.to_thread(serialize_payload_to_b64, my_payload)
                
                # 3. Send *our* payload and get theirs
                logger.info(f"Sending initiator payload to {state.peer_url}...")
                try:
                    response = await client.post(
                        "/exchange_weights",
                        json={
                            "agent_id": state.agent_id,
                            "payload_b64": my_payload_b64
                        }
                    )
                    response.raise_for_status()
                    
                    # 4. Deserialize *their* payload
                    logger.info("Received responder payload.")
                    response_data = response.json()
                    responder_payload = await asyncio.to_thread(
                        deserialize_payload_from_b64,
                        response_data["payload_b64"], state.global_model
                    )
                    if not responder_payload:
                        logger.error("Corrupt payload from peer. Skipping merge.")
                        continue

                except Exception as e:
                    logger.error(f"Failed to exchange with {state.peer_url}: {e}")
                    continue
                    
                # 5. Merge
                logger.info("Merging payloads...")
                merged_payload = await asyncio.to_thread(
                    merge_payloads, my_payload, responder_payload, 0.5
                )

                # 6. Update our global model
                await asyncio.to_thread(
                    update_global_model, state.global_model, merged_payload
                )
                
                # 7. Evaluate
                val_loss, val_acc, _, _ = await asyncio.to_thread(
                    evaluate, state.global_model, state.val_loader, state.device, state.criterion
                )
                logger.info(f"Round {state.round_num} Merged Accuracy: {val_acc:.2f}%")

# --- 4. Main Click Command to run BOTH ---
@click.command()
@click.option('--host', default='0.0.0.0', help="Host to bind the server to.")
@click.option('--port', default=8000, help="Port to bind the server to.")
@click.option('--data-client', default='client_0', help="Client data shard to use.")
@click.option('--peer-url', default='http://127.0.0.1:8001', help="URL of the peer agent to talk to.")
def main(host, port, data_client, peer_url):
    """
    Starts the Federated Learning Agent, running BOTH the
    Uvicorn server (Responder) and the Initiator loop.
    """
    agent_id = os.getenv("AGENT_NAME", f"agent-on-port-{port}")
    setup_logging(log_dir="logs", log_file=f"{agent_id}.log")
    
    # 1. Create the single shared state
    state = AgentState(
        agent_id=agent_id,
        peer_url=peer_url
    )
    state.load_data_and_model(
        data_client_id=data_client,
        val_client_id="test1" # All agents use the same validation setW
    )
    
    # 2. Attach the state to the FastAPI app
    app.state.agent_state = state
    
    # 3. Configure Uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    # 4. Run both the server and the initiator loop concurrently
    async def run_concurrently():
        await asyncio.gather(
            server.serve(),
            initiator_loop(state)
        )
    
    logger.info(f"Starting agent {agent_id} on {host}:{port}...")
    logger.info(f"Peer (active talker) URL set to: {peer_url}")
    asyncio.run(run_concurrently())

if __name__ == "__main__":
    main()