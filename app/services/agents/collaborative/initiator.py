import logging
import asyncio
import time
import httpx
import sys
import os

from langchain_core.messages import HumanMessage

# Local Imports
from services.agents.collaborative.state import AgentState, NUM_ROUNDS
from services.agents.collaborative.initiator_graph import initiator_graph

async def initiator_loop(state: AgentState, partner_urls: list):
    logging.info(f"--- {state.agent_id} | INITIATOR STARTING ---")
    
    # Populate state with available partners from config
    state.available_partners = partner_urls
    
    # Use explicit timeouts: 10m total, 1m connect for total round
    timeouts = httpx.Timeout(600.0, connect=60.0)
    
    # We create the httpx client ONCE here and share it via Singleton.
    # The 'select_partner' tool will use this to build A2A clients.
    for round_num in range(NUM_ROUNDS):
        logging.info(f"\n--- {state.agent_id} | Initiator Wakeup {round_num + 1}/{NUM_ROUNDS} ---")
        
        async with httpx.AsyncClient(timeout=timeouts) as httpx_client:
            state.shared_httpx_client = httpx_client

            state.active_client = None
            state.current_partner_id = None
            
            inputs = {"messages": [HumanMessage(content="Wake up. Evaluate status and execute a round.")]}
            
            try:
                logging.info("INITIATOR: Invoking Agent Brain...")
                await initiator_graph.ainvoke(inputs)
                logging.info("INITIATOR: Agent Brain finished processing.")
            except Exception as e:
                logging.error(f"INITIATOR: Graph execution failed: {e}", exc_info=True)
            
            # Cleanup Scratchpads
            state.initiator_working_weights = None
            state.initiator_incoming_payload = None
            
            # Wait before next wake up
            logging.info(f"INITIATOR: Round complete. Sleeping...")
            await asyncio.sleep(10)
    
    logging.info(f"--- {state.agent_id} | INITIATOR FINISHED ---")

def run_initiator_in_thread(state: AgentState, partner_urls: list):
    logging.info("INITIATOR: Starting initiator thread...")
    time.sleep(10) 
    try:
        asyncio.run(initiator_loop(state, partner_urls))
    except Exception as e:
        logging.error(f"INITIATOR: Initiator loop crashed: {e}", exc_info=True)