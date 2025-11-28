# Utility Functions Documentation

This reference covers the low-level utility functions available in `app/services/utils/`. You will use these functions when writing **new Tools** for your agent or modifying the existing graph logic.

These are the building blocks for **Training**, **Merging**, and **Communicating**.

---

## 1. Training & Evaluation
**Import:** `from utils.training import train, evaluate`

These functions handle the PyTorch training loops. They are designed to be stateless regarding the agent (you pass the model and data in).

### `train(...)`
The core learning loop. Supports **FedProx** regularization to prevent catastrophic forgetting.

```python
def train(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int, 
    learning_rate: float, 
    device: torch.device, 
    val_frequency: int = 1, 
    weight_decay: float = 0.0, 
    lr_scheduler_step_size: int = 7,
    global_model: nn.Module = None, 
    mu: float = 0.0, 
    log_prefix: str = "", 
    logger = None
) -> (nn.Module, list)
```

**Key Arguments:**
* **`model`**: The model instance to train. **Tip:** Always pass a `copy.deepcopy()` of your model, not the live reference, unless you want to modify it in place.
* **`global_model`** & **`mu`**: These control **FedProx**.
    * If `mu > 0.0` and `global_model` is provided, the loss function adds a penalty term: $\frac{\mu}{2} ||w - w_{global}||^2$.
    * **Strategy:** Use higher `mu` (e.g., 0.5 - 1.0) to stay close to a known good state. Use `mu=0.0` to learn new data aggressively.
* **`epochs`**: How many passes over the local data. **Warning:** High epochs (>2) can cause network timeouts for the partner waiting on you.

**Returns:**
1.  **`model`**: The trained model instance (modified in-place).
2.  **`history`**: A list of dicts containing loss/accuracy metrics per epoch.

---

### `evaluate(...)`
Runs a validation pass on the `test1` dataset.

```python
def evaluate(
    model: nn.Module, 
    val_loader: DataLoader, 
    device: torch.device, 
    criterion: nn.Module, 
    logger = None
) -> (float, float, int, int, int)
```

**Returns (5-Tuple):**
1.  **`avg_loss`** (float): CrossEntropy loss.
2.  **`accuracy`** (float): Overall accuracy percentage (0-100).
3.  **`correct`** (int): Raw count of correct predictions.
4.  **`total_samples`** (int): Total images in validation set.
5.  **`classes_learned`** (int): **Critical Metric.** The number of classes where the model achieved $\ge 10\%$ accuracy. Max is 40.
    * *Use this to detect if knowledge transfer is happening even if overall accuracy is stagnant.*

---

## 2. Federated Logic
**Import:** `from utils.federated_utils import merge_payloads`

### `merge_payloads(...)`
Mathematically combines two sets of weights. This is the "Federation" part.

```python
def merge_payloads(
    payload_A: dict, 
    payload_B: dict, 
    alpha: float
) -> dict
```

**Key Arguments:**
* **`payload_A`**: Usually your local weights (Draft).
* **`payload_B`**: Usually the partner's weights (Incoming).
* **`alpha`**: The weight given to `payload_A`.
    * **Formula:** $W_{new} = (\alpha \cdot A) + ((1 - \alpha) \cdot B)$
    * `alpha=0.5`: Balanced average (Standard FedAvg).
    * `alpha=0.8`: Trust your own data more (Conservative).
    * `alpha=0.2`: Trust the partner more (Aggressive adoption).

**Returns:**
* A new `state_dict` (dictionary of tensors) representing the merged model.

---

## 3. Payload Utilities
**Import:** `from utils.payload_utils import get_trainable_state_dict, serialize_payload_to_b64`

### `get_trainable_state_dict(...)`
Extracts *only* the weights that need to be shared.

```python
def get_trainable_state_dict(model: nn.Module) -> dict
```

* **Usage:** Call this on your model before serializing. It filters out frozen layers (like the pre-trained MobileNet backbone) to save bandwidth.

### `serialize_payload_to_b64(...)`
Compresses and encodes weights for network transmission.

```python
def serialize_payload_to_b64(state_dict: dict) -> str
```

* **Usage:** Must be called before creating a `WeightExchangePayload` object.
* **Returns:** A Base64 string representation of the weights.

---

## 4. Network & Communication
**Import:** `from utils.a2a_helpers import send_and_parse_a2a_message, WeightExchangePayload`

These helpers manage the A2A (Agent-to-Agent) protocol complexity.

### `WeightExchangePayload` (Class)
The data structure used for all messages.

```python
class WeightExchangePayload(BaseModel):
    agent_id: str       # Your ID
    payload_b64: str    # The weight string (or "" for text-only)
    message: str        # Chat message for the other agent
```

### `send_and_parse_a2a_message(...)`
**For Initiators Only.** Sends data to a partner and waits for their reply. Handles the timeout and parsing logic.

```python
async def send_and_parse_a2a_message(
    client: A2AClient, 
    request_payload: WeightExchangePayload, 
    reference_model: nn.Module, 
    logger = None
) -> (WeightExchangePayload, dict)
```

**Inputs:**
* **`client`**: The active connection object (from `state.active_client`).
* **`request_payload`**: The object containing your ID, message, and weights.
* **`reference_model`**: Required to validate the shape of the *incoming* weights to prevent corruption.

**Returns (Tuple):**
1.  **`response_data`**: The Pydantic object containing the partner's text message (`response_data.message`).
2.  **`responder_state_dict`**: The dictionary of weights sent by the partner.
    * **IMPORTANT:** This will be `None` if the partner sent a text-only reply (or if they signaled `BUSY`).
    * *Always check `if responder_state_dict:` before trying to merge.*
