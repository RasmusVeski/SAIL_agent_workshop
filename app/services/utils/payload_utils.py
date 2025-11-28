import torch
import torch.nn as nn
import logging
from io import BytesIO
import copy
import base64
import zlib


def get_trainable_state_dict(model: nn.Module) -> dict:
    """
    Extracts ONLY the parameters that require gradients into a new state dict.
    This can be saved over the network to avoid unnecessary latency.
    """
    trainable_state_dict = {}
    
    # Get the names of all parameters that are trainable
    trainable_names = {name for name, param in model.named_parameters() 
                         if param.requires_grad}
    
    if not trainable_names:
        logging.warning("get_trainable_state_dict: No trainable parameters found.")
        return {}

    # Create a new state dict containing only those parameters
    # We detach().clone() to avoid sending the entire computation graph
    full_state_dict = model.state_dict()
    for name, param in full_state_dict.items():
        if name in trainable_names:
            trainable_state_dict[name] = param.detach().clone()
            
    logging.debug(f"Extracted {len(trainable_state_dict)} trainable parameter groups.")

    return trainable_state_dict


def serialize_payload_to_b64(state_dict: dict) -> str:
    """
    Serializes a state dict into bytes, then Base64-encodes it into a string
    safe for JSON/HTTP transfer.
    """
    try:

        buffer = BytesIO()
        torch.save(state_dict, buffer)
        raw_bytes = buffer.getvalue()
        # Compress the bytes 
        compressed_bytes = zlib.compress(raw_bytes)
        # Encode to Base64 string
        payload_b64_string = base64.b64encode(compressed_bytes).decode('utf-8')
        
        # Optional: Log savings
        logging.debug(f"Compressed payload: {len(raw_bytes)} -> {len(compressed_bytes)} bytes")
        
        return payload_b64_string
    except Exception as e:
        logging.error(f"Failed to serialize payload: {e}")
        return None


def deserialize_payload_from_b64(payload_b64_string: str, reference_model: nn.Module) -> dict:
    """
    Decodes a Base64 string back into bytes, deserializes it into a
    state dict, and verifies its keys and shapes.
    """
    try:
        # Decode from Base64 string to compressed bytes
        compressed_bytes = base64.b64decode(payload_b64_string)
        
        raw_bytes = zlib.decompress(compressed_bytes) # Decompress 
        
        buffer = BytesIO(raw_bytes) # Load into Torch
        # Use weights_only=True for security 
        incoming_state_dict = torch.load(buffer, weights_only=True)

    except Exception as e:
        logging.error(f"Failed to deserialize payload: {e}", exc_info=True)
        return None

    if not isinstance(incoming_state_dict, dict):
        logging.error("Deserialized payload is not a dictionary.")
        return None
        
    # --- Verification Step ---
    reference_trainable_dict = get_trainable_state_dict(reference_model)
    
    # 1. Verify Keys
    if set(incoming_state_dict.keys()) != set(reference_trainable_dict.keys()):
        logging.error("Payload verification failed: Key mismatch.")
        logging.error(f"Expected: {set(reference_trainable_dict.keys())}")
        logging.error(f"Received: {set(incoming_state_dict.keys())}")
        return None
        
    # 2. Verify Shapes
    for key in reference_trainable_dict:
        expected_shape = reference_trainable_dict[key].shape
        received_shape = incoming_state_dict[key].shape
        if received_shape != expected_shape:
            logging.error(f"Payload verification failed: Shape mismatch for key '{key}'.")
            logging.error(f"Expected: {expected_shape}, Received: {received_shape}")
            return None
            
    logging.debug("Payload deserialized and verified successfully.")
    return incoming_state_dict