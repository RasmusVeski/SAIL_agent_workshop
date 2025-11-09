import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
import logging


def evaluate(model, val_loader, device, criterion):
    """
    Performs a single evaluation pass on the validation set.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in val_loader:
            if -1 in labels:
                continue
                
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0:
        logging.warning("Validation set empty or all batches bad.")
        return 0.0, 0.0

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_val_loss, accuracy, correct, total

def train(model, train_loader, val_loader, epochs, learning_rate, device, 
          val_frequency=1, weight_decay=0.0, lr_scheduler_step_size=7):
    """
    A standalone training function for an agent's model.
    Now includes weight_decay, a scheduler, and returns a metrics history.

    Parameters:
    - model (nn.Module): The model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - epochs (int): The number of epochs to train for.
    - learning_rate (float): The learning rate for the optimizer.
    - device (torch.device or str): The device to run training on ('cuda' or 'cpu').
    - val_frequency (int): Run validation every N epochs.
    - weight_decay (float): L2 regularization strength.
    - lr_scheduler_step_size (int): Step size for the learning rate scheduler.
    
    Returns:
    - model (nn.Module): The trained model.
    - history (list): A list of dictionaries containing metrics for each validation step.
    """
    
    history = []
    criterion = nn.CrossEntropyLoss()
    
    # Add weight_decay to the optimizer to combat overfitting
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Add a learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=0.1)
    
    model.to(device)
    
    # --- 1. Pre-training Evaluation ---
    if val_loader:
        logging.info("Performing pre-training evaluation...")
        avg_val_loss, accuracy, correct, total = evaluate(model, val_loader, device, criterion)
        logging.info(f"--- PRE-TRAINING VALIDATION ---")
        logging.info(f"Validation Loss: {avg_val_loss:.4f} | "
                     f"Validation Acc: {accuracy:.2f}% ({correct}/{total})")
        history.append({
            'epoch': 0,
            'train_loss': None,
            'val_loss': avg_val_loss,
            'val_acc': accuracy
        })
    
    logging.info(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        # --- 2. Training Phase ---
        model.train()
        running_loss = 0.0
        epoch_train_loss = 0.0
        train_batches = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for i, (data, labels) in enumerate(train_loader_tqdm):
            if -1 in labels:
                logging.warning("Skipping bad batch")
                continue
                
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            item_loss = loss.item()
            running_loss += item_loss
            epoch_train_loss += item_loss
            train_batches += 1
            
            if i % 10 == 9: 
                avg_loss = running_loss / 10
                train_loader_tqdm.set_postfix(loss=f"{avg_loss:.3f}")
                logging.debug(f"Epoch {epoch+1}, Batch {i+1}, Avg. Loss: {avg_loss:.3f}")
                running_loss = 0.0
        
        # Step the learning rate scheduler
        scheduler.step()

        avg_epoch_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0.0
        logging.info(f"Epoch {epoch+1}/{epochs} training complete. Avg Train Loss: {avg_epoch_train_loss:.4f}")

        # --- 3. Validation Phase ---
        if val_loader and (epoch + 1) % val_frequency == 0:
            avg_val_loss, accuracy, correct, total = evaluate(model, val_loader, device, criterion)
            
            logging.info(f"--- Epoch {epoch+1}/{epochs} VALIDATION ---")
            logging.info(f"Validation Loss: {avg_val_loss:.4f} | "
                         f"Validation Acc: {accuracy:.2f}% ({correct}/{total})")
            history.append({
                'epoch': epoch + 1,
                'train_loss': avg_epoch_train_loss,
                'val_loss': avg_val_loss,
                'val_acc': accuracy
            })
        
        elif val_loader and (epoch + 1) % val_frequency != 0:
             logging.info(f"Epoch {epoch+1}/{epochs} training complete (validation skipped).")
        
        elif not val_loader:
             logging.info(f"Epoch {epoch+1}/{epochs} training complete (no validation).")

    logging.info("Finished Training")
    return model, history