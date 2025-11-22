import logging
import copy
import asyncio
import sys
import os
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, DataPart
from a2a.utils import new_agent_text_message

# Local Imports
from services.agents.without_llm.state import AgentState, thread_safe_merge_and_evaluate

# Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.training import train
from utils.payload_utils import get_trainable_state_dict, serialize_payload_to_b64
from utils.a2a_helpers import (
    WeightExchangePayload,
    parse_incoming_request,
    send_a2a_response
)

class FederatedAgentExecutor(AgentExecutor):
    
    def __init__(self, state: AgentState):
        self.state = state

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        logging.info(f"--- {self.state.agent_id} | RESPONDER ---")
        logging.info("RESPONDER: Received federated_weight_exchange request.")
        
        # 1. Deserialize initiator's payload
        try:
            request_data, initiator_payload = parse_incoming_request(
                context, self.state.global_model
            )
            logging.info(f"RESPONDER: Message from {request_data.agent_id}: {request_data.message}")
        except Exception as e:
            logging.error(f"RESPONDER: Payload deserialization failed: {e}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: Corrupt payload: {e}")
            )
            return
        
        def _safe_copy_model(state):
            with state.model_lock:
                return copy.deepcopy(state.global_model)
        
        # 2. Train (Async/Threaded)
        logging.info("RESPONDER: Copying global model for responder training...")
        local_model_to_train = await asyncio.to_thread(_safe_copy_model, self.state)

        logging.info("RESPONDER: Training local model (as responder)...")
        local_model, _ = await asyncio.to_thread(
            train,
            model=local_model_to_train,
            train_loader=self.state.train_loader,
            val_loader=None,
            global_model=local_model_to_train,
            device=self.state.device,
            **self.state.agent_hps
        )

        # 3. Prepare payload
        my_payload = await asyncio.to_thread(get_trainable_state_dict, local_model)
        my_payload_b64 = await asyncio.to_thread(serialize_payload_to_b64, my_payload)

        # 4. Send response
        response_payload = WeightExchangePayload(
            agent_id=self.state.agent_id,
            payload_b64=my_payload_b64,
            message="Here is my updated model!"
        )
        
        try:
            await send_a2a_response(event_queue, response_payload)
        except Exception as e:
            logging.error(f"RESPONDER: Failed to send response: {e}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: Failed to build response: {e}")
            )
            return
            
        # 5. Merge (3-Way)
        await thread_safe_merge_and_evaluate(
            state=self.state,
            payload_1=my_payload,
            payload_2=initiator_payload,
            role="Responder"
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        logging.warning("RESPONDER: Cancel not supported")
        await event_queue.enqueue_event(new_agent_text_message("Error: Cancel not supported"))