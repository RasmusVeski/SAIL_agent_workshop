import os
import sys
import torch
import logging
from dotenv import load_dotenv

# --- 1. Path Setup (To import from utils) ---
# We need to go up two levels from services/agents/ to reach app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model import FoodClassifier
from utils.data_loader import create_dataloader
from utils.training import train, evaluate

# --- 2. Configuration ---
load_dotenv()

# We use a standard logger here (cyan/green not needed for single process)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BENCHMARK] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# Benchmark Settings (Training longer to find true convergence)
BENCHMARK_HPS = {
    'epochs': 5,           # Train longer to see max potential
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'val_frequency': 1,
    'lr_scheduler_step_size': 10
}

def main():
    agent_id = os.getenv("AGENT_ID", "Benchmark_Agent")
    base_dir = os.getenv("BASE_DATA_DIR", "./app/sharded_data")
    data_dir_name = os.getenv("DATA_DIR_NAME")
    val_dir_name = os.getenv("VAL_DIR_NAME", "test1")

    # Paths
    train_path = os.path.join(base_dir, data_dir_name)
    val_path = os.path.join(base_dir, val_dir_name)
    
    logging.info(f"--- STARTING LOCAL BENCHMARK FOR {agent_id} ---")
    logging.info(f"Train Data: {train_path}")
    logging.info(f"Val Data:   {val_path}")

    # 1. Load Data
    train_loader = create_dataloader(train_path, batch_size=32, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_path, batch_size=32, shuffle=False, num_workers=0)

    if not train_loader or not val_loader:
        logging.error("Could not create dataloaders. Exiting.")
        return

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoodClassifier()
    
    # 3. Initial Eval
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
    logging.info(f"Initial (Pre-train) Accuracy: {val_acc:.2f}%")

    # 4. Train (Using the shared training utility)
    # Note: We pass mu=0.0 because there is no global model to constrain us (no FedProx)
    logging.info("Starting training...")
    trained_model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        global_model=None, # No FedProx
        mu=0.0,            # No FedProx
        **BENCHMARK_HPS
    )

    # 5. Final Results
    final_acc = next(
    (h['val_acc'] for h in reversed(history) if h['val_acc'] is not None),
    None
    )
    vals = [h['val_acc'] for h in history if h['val_acc'] is not None]
    best_acc = max(vals) if vals else None
        
    logging.info("-" * 30)
    logging.info(f"BENCHMARK COMPLETE")
    if final_acc is None:
        logging.info(f"Final Accuracy: N/A (no validation for last epoch)")
    else:
        logging.info(f"Final Accuracy: {final_acc:.2f}%")
    if best_acc is None:
        logging.info(f"Best Accuracy: N/A")
    else:
        logging.info(f"Peak Accuracy:  {best_acc:.2f}%")
    logging.info("-" * 30)

    # 6. Save
    save_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"benchmark_{agent_id}.pt")
    torch.save(trained_model.state_dict(), save_path)
    logging.info(f"Saved benchmark model to {save_path}")

if __name__ == "__main__":
    # Limit threads to prevent CPU hogging if run alongside agents
    torch.set_num_threads(4)
    main()