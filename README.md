# SAIL Collaborative Agents Workshop

This repository allows simulation of a **Decentralized Federated Learning** network where nodes are controlled by Autonomous Agents (LLMs).


---

## Setting up

### 1. Prerequisites
* **Docker:** https://docs.docker.com/engine/install/
  * Works also uncontainerized but good practice to put agents in containers
* **OpenAI API Key:** You need a valid API key for a 
* **Mutual subnet** The machines this experiment is running on need to be on the same virtual net, [NetBird](https://app.netbird.io/) works.

### 2. Installation
Clone the repository and navigate to the project root:

```bash
git clone https://github.com/RasmusVeski/SAIL_agent_hackathon.git
cd SAIL_agent_hackathon
```

Create a `.env` file in the root directory. You can copy use the .env.example as template or the example below:

```bash
# .env file
AGENT_ID=Agent_A
PORT=9000
PUBLIC_URL=http://xxx.yy.zz.x:9000 #your machine address and port
# List of partner machine addresses running simulation with
PARTNER_URLS=http://yyy.xx.zzz.x:9000,http://zzz.xx.yy.x:9000 
# Dataset dependent
BASE_DATA_DIR=/app/sharded_data
DATA_DIR_NAME=client_0 #your shard name
VAL_DIR_NAME=test1
OPENAI_API_BASE=https://inference.rcp.epfl.ch/v1
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL_NAME=openai/gpt-oss-120b
```

### 3. Data
This workshop is built using the [Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101). The data is split into multiple non-IID shards and will be distributed during the workshop.
Place the training data in ``/app/sharded_data/client_x`` and testing data in ``/app/sharded_data/test1``.

### 4. Building the Environment
Build the image **once**:

```bash
docker build -t agent-image .
```

### 5. Baselines-Data exploration

* **Discover local data split**
    * *Goal:* See how what types of classes and how many samples you have.
    ```bash
    docker run -it --rm \
        --name data_check \
        -v ./app:/app \
        -v ./app/sharded_data:/app/sharded_data \
        --env-file ./.env \
        agent-image:latest \
        analyze_local_data.py
    ```

* **Run the Local Benchmark (Baseline)**
    * *Goal:* See how well the model performs purely on local data.
    ```bash
    docker run -it --rm \
        --name local_benchmark \
        --privileged \
        -v ./app:/app \
        -v ./app/sharded_data:/app/sharded_data \
        -v ./logs:/app/logs \
        --env-file ./.env \
        agent-image:latest \
        services/agents/benchmark_local.py
    ```

* **Run the No LLM Agent Benchmark**
    * *Goal:* A deterministic script for testing infrastructure without LLM unpredictability.
    ```bash
    docker run -it --rm \
        --name agent_benchmark \
        --privileged \
        -v ./app:/app \
        -v ./app/sharded_data:/app/sharded_data \
        -v ./logs:/app/logs \
        --env-file ./.env \
        agent-image:latest \
        services/agents/without_llm/main.py
    ```


### 6. Experiments
We use Docker's `ENTRYPOINT` pattern. You can switch between different agent modes by appending the script path to the run command.

* **Run the Collaborative Agent (LLM-Driven)**
    * *Goal:* Agents cooperate to improve accuracy.
    ```bash
    docker run -it --rm \
        --name agent_collaborative \
        --privileged \
        -v ./app:/app \
        -v ./app/sharded_data:/app/sharded_data \
        -v ./logs:/app/logs \
        --env-file ./.env \
        agent-image:latest \
        services/agents/collaborative/main.py
    ```

* **Run the Competitive Agent (LLM-Driven)**
    * *Goal:* Agents try to only to be better than other agents
    ```bash
    docker run -it --rm \
        --name agent_competitive \
        --privileged \
        -v ./app:/app \
        -v ./app/sharded_data:/app/sharded_data \
        -v ./logs:/app/logs \
        --env-file ./.env \
        agent-image:latest \
        services/agents/competitive/main.py
    ```

---

## Project Structure & Files

This workshop is divided into two halves. First we will build agents that try to get the best model, then we will train agents who simply need a better model than others.

### Files you will edit during the workshop 

#### 1. Collaborative Agents (`app/services/agents/collaborative/`)
* **`initiator_graph.py`**: Defines the **Initiator** workflow (Proactive). Edit this to change how the agent selects partners, decides to train, or interprets results.
* **`responder_graph.py`**: Defines the **Responder** workflow (Reactive). Edit this to change how the agent reacts to incoming requests.
* **`state.py`**: Holds the global state (Model weights, History, Locks). Edit this to change default Hyperparameters (`AGENT_HPS`) or to add new assisting global parameters and logic.


#### 2. Competitive Agents (`app/services/agents/competitive/`)
* *(Future/TODO)*: Agents designed to compete, hide data, or potentially poison the model.

### The Infrastructure (Utils)
These files handle the heavy lifting. You generally **do not** need to edit these unless you are changing the core learning algorithm.

* **`app/services/utils/`**:
    * `training.py`: Contains the PyTorch training loop and evaluation logic.
    * `federated_utils.py`: Handles merging weights (`FedAvg` logic).
    * `a2a_helpers.py`: Manages sending messages between two agents using A2A.
    * `logger_colored.py`: Provides the beautiful color-coded logs for debugging.
    * `payload_utils.py`: Logic for extracting and serializing trainable state dicts from the trained model.

---

## Building your agents: important notes

### Given tools
In both `initiator_graph.py` and `responder_graph.py` you will already see a couple tools implemented. Read their docstring to understand their purpose. These tools are tied to the LLM and the primary way you can improve your agents is making more tools or editing existing ones.

### Important variables
In `state.py` you will see many global variables. Read the comment next to 

### Agent node
