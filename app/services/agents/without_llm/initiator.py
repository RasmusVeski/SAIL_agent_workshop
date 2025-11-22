import logging
import copy
import asyncio
import time
import httpx
import sys
import os

from a2a.client import A2ACardResolver, ClientFactory, ClientConfig, A2AClient

# Local Imports
from services.agents.without_llm.state import AgentState, NUM_ROUNDS, thread_safe_merge_and_evaluate

# Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.training import train
from utils.payload_utils import get_trainable_state_dict, serialize_payload_to_b64
from utils.a2a_helpers import WeightExchangePayload, send_and_parse_a2a_message

async def get_partner_client(httpx_client: httpx.AsyncClient, partner_url: str) -> A2AClient | None:
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=partner_url)
    for attempt in range(5):
        try:
            logging.info(f"INITIATOR: Attempting to resolve partner card at {partner_url}...")
            partner_card = await resolver.get_agent_card()
            logging.info(f"INITIATOR: Successfully resolved partner: {partner_card.name}")
            config = ClientConfig(httpx_client=httpx_client)
            return await ClientFactory.connect(agent=partner_card, client_config=config)
        except Exception as e:
            logging.warning(f"INITIATOR: Failed to resolve partner card (attempt {attempt+1}/5): {e}")
            await asyncio.sleep(10)
    logging.error("INITIATOR: Could not resolve partner agent card. Initiator loop will not run.")
    return None

async def initiator_loop(state: AgentState, partner_url: str):
    logging.info(f"INITIATOR: --- {state.agent_id} | INITIATOR ---")
    logging.info(f"INITIATOR: Starting initiator loop. Will federate with {partner_url}")
    
    # Use explicit timeouts: 10m total, 1m connect
    timeouts = httpx.Timeout(600.0, connect=60.0)
    async with httpx.AsyncClient(timeout=timeouts) as httpx_client:
        
        client = await get_partner_client(httpx_client, partner_url)
        if not client:
            return 

        for round_num in range(NUM_ROUNDS):
            logging.info(f"INITIATOR: \n--- {state.agent_id} | Initiator Round {state.round_num}/{NUM_ROUNDS} ---")
        
            def _safe_copy_model(state):
                with state.model_lock:
                    return copy.deepcopy(state.global_model)
            
            logging.info("INITIATOR: Training local model (as initiator)...")
            current_global_model = await asyncio.to_thread(_safe_copy_model, state)
            
            # Give network a tiny breath
            await asyncio.sleep(0.1)
            
            local_model, _ = await asyncio.to_thread(
                train,
                model=current_global_model,
                train_loader=state.train_loader,
                val_loader=None,
                global_model=current_global_model,
                device=state.device,
                **state.agent_hps
            )

            my_payload = await asyncio.to_thread(get_trainable_state_dict, local_model)
            my_payload_b64 = await asyncio.to_thread(serialize_payload_to_b64, my_payload)
            request_payload_obj = WeightExchangePayload(
                agent_id=state.agent_id,
                payload_b64=my_payload_b64,
                message=f"Starting round {state.round_num}"
            )

            logging.info("INITIATOR: Sending initiator payload to partner...")
            try:
                response_data, responder_payload = await send_and_parse_a2a_message(
                    client, request_payload_obj, state.global_model
                )
                logging.info(f"INITIATOR: Message from {response_data.agent_id}: {response_data.message}")
                
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
        
            logging.info(f"Round {state.round_num} complete. Waiting 30s...")
            await asyncio.sleep(30)
    
    logging.info(f"--- {state.agent_id} | INITIATOR FINISHED ---")

def run_initiator_in_thread(state: AgentState, partner_url: str):
    logging.info("INITIATOR: Starting initiator thread...")
    time.sleep(10) #Give server time to start
    try:
        asyncio.run(initiator_loop(state, partner_url))
    except Exception as e:
        logging.error(f"INITIATOR: Initiator loop crashed: {e}", exc_info=True)