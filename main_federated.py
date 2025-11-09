import torch
import os
import logging
import copy
import pprint
from utils.model import FoodClassifier
from utils.data_loader import create_dataloader
from utils.training import train, evaluate
from utils.logger_setup import setup_logging
from utils.federated_utils import merge_models

def main():
    """
    Main function to simulate a simple 2-client federated learning round
    using the new "agent-driven" partial merge.
    """
    # --- 1. Setup ---
    setup_logging(log_dir="logs", log_file="federated_training.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device: {device}")

    # --- 2. Define Paths ---
    BASE_DIR = "./sharded_data"
    CLIENT_0_DIR = os.path.join(BASE_DIR, "client_0")
    CLIENT_1_DIR = os.path.join(BASE_DIR, "client_1")
    VAL_DIR = os.path.join(BASE_DIR, "test1")

    # --- 3. Hyperparameters ---
    EPOCHS_PER_ROUND = 3 
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    LR_SCHEDULER_STEP = 5
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    # --- 4. Load Data ---
    logging.info(f"Loading data for Client 0 ({CLIENT_0_DIR})")
    train_loader_0 = create_dataloader(CLIENT_0_DIR, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    logging.info(f"Loading data for Client 1 ({CLIENT_1_DIR})")
    train_loader_1 = create_dataloader(CLIENT_1_DIR, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    logging.info(f"Loading validation data ({VAL_DIR})")
    val_loader = create_dataloader(VAL_DIR, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    if not train_loader_0 or not train_loader_1 or not val_loader:
        logging.error("Failed to load all dataloaders. Exiting.")
        return

    # --- 5. Initialize Models ---
    # Both clients start with the *same* initial model
    model_0 = FoodClassifier()
    model_1 = copy.deepcopy(model_0)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # --- 6. Pre-Training Evaluation ---
    logging.info("--- Evaluating Base Model (Round 0) ---")
    val_loss, val_acc, _, _ = evaluate(model_0, val_loader, device, criterion)
    logging.info(f"Round 0 Accuracy: {val_acc:.2f}%")
    baseline_acc = val_acc

    # --- 7. FEDERATED ROUND 1 ---
    logging.info("\n--- STARTING FEDERATED ROUND 1 ---")
    
    # Client 0 trains locally
    logging.info("--- Training Client 0 ---")
    model_0, history_0 = train(
        model=model_0,
        train_loader=train_loader_0,
        val_loader=val_loader, 
        epochs=EPOCHS_PER_ROUND,
        learning_rate=LEARNING_RATE,
        device=device,
        val_frequency=EPOCHS_PER_ROUND,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_step_size=LR_SCHEDULER_STEP
    )
    
    # Client 1 trains locally
    logging.info("--- Training Client 1 ---")
    model_1, history_1 = train(
        model=model_1,
        train_loader=train_loader_1,
        val_loader=val_loader,
        epochs=EPOCHS_PER_ROUND,
        learning_rate=LEARNING_RATE,
        device=device,
        val_frequency=EPOCHS_PER_ROUND,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_step_size=LR_SCHEDULER_STEP
    )

    acc_0 = history_0[-1]['val_acc']
    acc_1 = history_1[-1]['val_acc']
    logging.info(f"Client 0 final acc: {acc_0:.2f}%")
    logging.info(f"Client 1 final acc: {acc_1:.2f}%")

    # --- 8. Merge Step (The "Agent" Logic) ---
    logging.info("\n--- MERGING MODELS (Agent Logic) ---")
    
    # This is the "hyperparameter" your agent would choose.
    # For now, we simulate a simple 50/50 average.
    # A smart agent might choose alpha=0.9 if acc_0 > acc_1.
    # A competitive agent might choose alpha=0.0 to steal all of model_1's weights.
    
    merge_alpha = 0.5 
    logging.info(f"Agent strategy: merging with alpha = {merge_alpha}")
    
    # We merge Client 0's model with Client 1's model.
    global_model = merge_models(model_0, model_1, alpha=merge_alpha)

    # --- 9. Evaluate Merged Model ---
    logging.info("--- Evaluating Merged Global Model ---")
    merged_val_loss, merged_val_acc, _, _ = evaluate(global_model, val_loader, device, criterion)
    
    logging.info(f"Baseline (Round 0) Accuracy: {baseline_acc:.2f}%")
    logging.info(f"Merged Global (Round 1) Accuracy: {merged_val_acc:.2f}%")
    
    logging.info("Federated simulation finished.")


if __name__ == "__main__":
    main()