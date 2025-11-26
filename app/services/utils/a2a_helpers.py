# app/services/utils/a2a_helpers.py

import logging
import torch
import torch.nn as nn
from pydantic import BaseModel
from uuid import uuid4

# --- A2A Imports ---
from a2a.client import A2AClient
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, DataPart, Task

# --- Utility Imports ---
# This assumes a2a_helpers.py is in the same 'utils' folder as payload_utils.py
from .payload_utils import deserialize_payload_from_b64


# --- Shared Pydantic Model ---
# We define this here so both main.py and the helpers can import it
class WeightExchangePayload(BaseModel):
    agent_id: str
    payload_b64: str = ""
    message: str

def _get_logger(logger=None):
    """Helper to ensure we always have a logger instance."""
    return logger if logger else logging.getLogger()

# --- Helper 1: Parse Incoming Requests ---
def parse_incoming_request(context: RequestContext, reference_model: nn.Module, logger=None) -> (WeightExchangePayload, dict):
    """
    Parses an incoming A2A request context to extract and deserialize
    the WeightExchangePayload and the model state_dict.
    Raises ValueError on failure.
    """

    log = _get_logger(logger)

    if not context.message.parts:
        raise ValueError("Request message has no parts")
    
    part_root = context.message.parts[0].root
    if part_root.kind != 'data':
        raise ValueError(f"Expected part kind 'data', but got '{part_root.kind}'")
        
    # 1. Deserialize the Pydantic model
    request_data = WeightExchangePayload(**part_root.data)
    
    # 2. Check if there is a payload to deserialize
    payload_state_dict = None
    if request_data.payload_b64:
        payload_state_dict = deserialize_payload_from_b64(
            request_data.payload_b64, reference_model
        )
        if not payload_state_dict:
            log.warning(f"RESPONDER: Warning - Payload present but failed to deserialize from {request_data.agent_id}")
    else:
        log.info(f"RESPONDER: Received text-only message from {request_data.agent_id}: {request_data.message}")
    return request_data, payload_state_dict


# --- Helper 2: Send a Response ---
async def send_a2a_response(event_queue: EventQueue, response_payload: WeightExchangePayload, logger=None):
    """
    Builds and enqueues an A2A response message.
    """
    log = _get_logger(logger)

    try:
        response_data_part = DataPart(data=response_payload.model_dump())
        response_part = Part(root=response_data_part)
        response_message = Message(
            role='agent',
            parts=[response_part],
            messageId=uuid4().hex
        )
        await event_queue.enqueue_event(response_message)
        log.info(f"RESPONDER: Sent responder payload to agent {response_payload.agent_id}.")
    except Exception as e:
        # Re-raise to be caught by the executor
        raise ValueError(f"Failed to build or enqueue response message: {e}")


# --- Helper 3: Send a Message and Parse Response ---
async def send_and_parse_a2a_message(client: A2AClient, 
                                     request_payload: WeightExchangePayload, 
                                     reference_model: nn.Module,
                                     logger=None) -> (WeightExchangePayload, dict):
    """
    Sends an A2A message, parses the complex response, and returns
    the deserialized WeightExchangePayload and model state_dict.
    Raises exceptions on failure.
    """
    log = _get_logger(logger)

    target_name = client._card.name if hasattr(client, '_card') else "Unknown Partner"
    log.info(f"INITIATOR: Sending initiator payload from {request_payload.agent_id} to {target_name}...")
    
    # 1. Build request
    message_part_data = {
        "kind": "data",
        "data": request_payload.model_dump()
    }
    message_to_send = Message(
        role="user",
        parts=[message_part_data],
        messageId=uuid4().hex
    )

    # 2. Send and parse generator response
    response_generator = client.send_message(message_to_send)
    response_item = None
    async for item in response_generator:
        response_item = item
        break # We expect only one item

    if response_item is None:
        raise Exception("Agent did not return a response")

    response_message: Message | None = None
    if isinstance(response_item, Message):
        response_message = response_item
    elif isinstance(response_item, tuple) and isinstance(response_item[0], Task):
        task: Task = response_item[0]
        if task.history:
            response_message = task.history[-1]
        else:
            raise Exception("Received a Task object with no history")
    else:
        raise Exception(f"Received unexpected response type: {type(response_item)}")
    
    # 3. Parse the A2A Message object
    if not response_message.parts:
        raise ValueError("Response message has no parts")
    
    part_root = response_message.parts[0].root
    if part_root.kind != 'data':
        raise ValueError(f"Response part is not 'data', got {part_root.kind}")
    
    # 4. Deserialize the Pydantic model and the b64 payload
    response_data = WeightExchangePayload(**part_root.data)
    
    # Case A: The Partner is BUSY
    if response_data.message == "BUSY":
        log.info(f"INITIATOR: Partner {response_data.agent_id} signaled BUSY.")
        return response_data, None

    # Case B: The Partner sent weights
    responder_state_dict = None
    if response_data.payload_b64:
        responder_state_dict = deserialize_payload_from_b64(
            response_data.payload_b64, reference_model
        )
        if not responder_state_dict:
             # Only raise error if payload WAS provided but failed
            raise ValueError("Received corrupt payload from responder")
            
        log.info(f"INITIATOR: Received valid weights from {response_data.agent_id}")
    else:
        # Case C: Text-only response (e.g., just a chat message)
        log.info(f"INITIATOR: Received text-only response from {response_data.agent_id}")

    return response_data, responder_state_dict