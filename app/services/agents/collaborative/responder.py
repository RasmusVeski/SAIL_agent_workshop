import logging
import asyncio
import sys
import os
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

# Local Imports
from services.agents.collaborative.state import AgentState, state_singleton
from services.agents.collaborative.responder_graph import responder_graph

# Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.payload_utils import serialize_payload_to_b64, get_trainable_state_dict
from utils.a2a_helpers import (
    WeightExchangePayload,
    parse_incoming_request,
    send_a2a_response
)

class CollaborativeResponderExecutor(AgentExecutor):
    
    def __init__(self, state: AgentState):
        self.state = state

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        
        # --- 1. BUSY CHECK (Fast Fail) ---
        if self.state.responder_lock.locked():
            logging.info("RESPONDER: Locked/Busy. Rejecting incoming request immediately.")
            
            # Send a polite "Busy" payload
            # We send None/Empty string for payload to save bandwidth
            busy_payload = WeightExchangePayload(
                agent_id=self.state.agent_id,
                payload_b64="", # Empty payload
                message="BUSY"  # Magic keyword for the initiator to recognize
            )
            
            await send_a2a_response(event_queue, busy_payload)
            return
        
        # --- CRITICAL: ACQUIRE LOCK ---
        # This ensures we don't overwrite the singleton if multiple agents
        async with self.state.responder_lock:

            logging.info(f"--- {self.state.agent_id} | RESPONDER ACTIVE ---")
            logging.info("RESPONDER: Received federated_weight_exchange request.")
            
            # 1. Deserialize initiator's payload
            try:
                request_data, initiator_payload = parse_incoming_request(
                    context, self.state.global_model
                )
                logging.info(f"RESPONDER: Message from {request_data.agent_id}: {request_data.message}")

                self.state.responder_incoming_payload = initiator_payload
                self.state.responder_working_weights = None
            except Exception as e:
                logging.error(f"RESPONDER: Payload deserialization failed: {e}")
                await event_queue.enqueue_event(
                    new_agent_text_message(f"RESPONDER: Error: Corrupt payload: {e}")
                )
                return
            
            initial_msg = f"Partner {request_data.agent_id} sent weights. Message: {request_data.message}"
            inputs = {
                "messages": [HumanMessage(content=initial_msg)],
                "partner_id": request_data.agent_id 
            }

            try:
                logging.info("RESPONDER: Invoking Agent Brain...")
                final_state = await responder_graph.ainvoke(inputs)
                logging.info("RESPONDER: Agent Brain finished processing.")
            except Exception as e:
                logging.error(f"RESPONDER: Graph execution failed: {e}. Proceeding with fallback.")
                final_state = {}


            weights_to_send = None
            response_msg = "Ready for next round!"

            if final_state and "messages" in final_state and final_state["messages"]:
                last_msg = final_state["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    response_msg = str(last_msg.content)
                    logging.info(f"RESPONDER: Agent chose to say: '{response_msg}'")

            # Check for draft weights (Training happened, but no commit)
            if state_singleton.responder_working_weights is not None:
                logging.info("RESPONDER: Found draft (newest) weights from Agent.")
                weights_to_send = state_singleton.responder_working_weights
                state_singleton.responder_working_weights = None
            else:
                # Fallback: Send Current Global (Either Commit happened OR Nothing happened)
                def _get_safe_current_weights():
                    with self.state.model_lock:
                        return get_trainable_state_dict(self.state.global_model)
                
                weights_to_send = await asyncio.to_thread(_get_safe_current_weights)
                

            # Serialize and Send
            # We use asyncio.to_thread because serialization can be slow
            my_payload_b64 = await asyncio.to_thread(
                serialize_payload_to_b64, 
                weights_to_send
            )
            response_payload = WeightExchangePayload(
                agent_id=self.state.agent_id,
                payload_b64=my_payload_b64,
                message=response_msg
            )

            
            try:
                await send_a2a_response(event_queue, response_payload)
            except Exception as e:
                logging.error(f"RESPONDER: Failed to send response: {e}")
                await event_queue.enqueue_event(
                    new_agent_text_message(f"Error: Failed to build response: {e}")
                )

            self.state.responder_incoming_payload = None
            self.state.responder_working_weights = None
            logging.info(f"--- {self.state.agent_id} | RESPONDER DONE ---")


    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        logging.warning("Cancel not supported")
        await event_queue.enqueue_event(new_agent_text_message("Error: Cancel not supported"))