import torch
import os
import logging
from utils.model import FoodClassifier
from utils.data_loader import create_dataloader
from utils.training import train
from utils.logger_setup import setup_logging
import pprint

def main():
    """
    Main function to run the training pipeline.
    """
    # --- 1. Setup Logging ---
    setup_logging(log_dir="logs", log_file="main_training.log")

    # --- 2. Define Paths and Hyperparameters ---
    logging.info("Setting up configuration...")
    
    # Paths
    BASE_DIR = "../setup/sharded_data"
    #BASE_DIR = "./sharded_data"
    #TRAIN_DATA_DIR = os.path.join(BASE_DIR, "client_0")
    TRAIN_DATA_DIR = os.path.join(BASE_DIR, "train")
    VAL_DATA_DIR = os.path.join(BASE_DIR, "test1")

    # Hyperparameters
    EPOCHS = 15
    VAL_FREQUENCY = 3  
    NUM_WORKERS = 0 
    BATCH_SIZE = 32
    
    # --- NEW Hyperparameters for Fine-Tuning ---
    # Fine-tuning requires a smaller learning rate to avoid
    # destroying the pre-trained weights.
    LEARNING_RATE = 1e-4  # Was 0.001
    
    # Increase regularization to compensate for more trainable parameters
    WEIGHT_DECAY = 1e-3 # Was 1e-4
    
    # Let's shorten the scheduler step since LR is already low
    LR_SCHEDULER_STEP = 5 

    # --- 3. Set Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Running on device: {device}")
    logging.info(f"Training data: {TRAIN_DATA_DIR}")
    logging.info(f"Validation data: {VAL_DATA_DIR}")

    # --- 4. Create DataLoaders ---
    logging.info("Creating training dataloader...")
    train_loader = create_dataloader(
        data_dir=TRAIN_DATA_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    logging.info("Creating validation dataloader...")
    val_loader = create_dataloader(
        data_dir=VAL_DATA_DIR,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    if not train_loader or not val_loader:
        logging.error("Error creating dataloaders. Exiting.")
        return

    logging.info(f"Loaded {len(train_loader.dataset)} training samples.")
    logging.info(f"Loaded {len(val_loader.dataset)} validation samples.")

    # --- 5. Initialize Model ---
    model = FoodClassifier()
    
    logging.info(f"Model initialized: {model.__class__.__name__}")
    logging.info(f"Total parameters (all):       {model.count_params(only_trainable=False)}")
    logging.info(f"Total parameters (trainable): {model.count_params(only_trainable=True)}")

    # --- 6. Run Training ---
    try:
        model, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            device=device,
            val_frequency=VAL_FREQUENCY,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_step_size=LR_SCHEDULER_STEP
        )
        
        logging.info("--- TRAINING HISTORY ---")
        logging.info(f"\n{pprint.pformat(history)}")
        
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
    
    logging.info("Main script finished.")


if __name__ == "__main__":
    main()