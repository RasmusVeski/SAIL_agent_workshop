import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
import logging
import numpy as np

NUM_CLASSES = 40 #For Food dataset

def evaluate(model, val_loader, device, criterion, logger=None):
    """
    Performs a single evaluation pass on the validation set.
    """

    # Fallback to root logger if specific logger not provided
    log = logger if logger else logging.getLogger()

    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    valid_batches = 0

    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))
    
    with torch.no_grad():
        for data, labels in val_loader:
            if (labels == -1).any():
                continue

            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            valid_batches += 1

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class calculation
            c = (predicted == labels).squeeze()
            label_cpu = labels.cpu()
            c_cpu = c.cpu()

            for i in range(labels.size(0)):
                label = label_cpu[i].item()
                if label < NUM_CLASSES: # Safety check
                    class_correct[label] += c_cpu[i].item()
                    class_total[label] += 1

    if valid_batches == 0 or total_samples == 0:
        log.warning("Validation had zero valid batches.")
        return 0.0, 0.0, 0, 0

    avg_loss = total_loss / valid_batches
    overall_accuracy = 100 * correct / total_samples

    # --- NEW: Log Per-Class Stats ---
    # Calculate accuracy for each class that actually appeared in the validation set
    accuracies = []
    classes_with_data = 0
    
    # log.info("--- Per-Class Performance ---") # Optional: Uncomment for verbose logs
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            accuracies.append(acc)
            classes_with_data += 1
            # log.info(f"Class {i}: {acc:.1f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            accuracies.append(0.0)
    macro_avg_acc = np.mean(accuracies)
    # Count how many classes have > 0% accuracy (Knowledge Width)
    classes_learned = sum(1 for acc in accuracies if acc > 0.0)
    log.info(f"[Eval Details] Macro-Avg Acc: {macro_avg_acc:.2f}% | Classes Learned: {classes_learned}/{NUM_CLASSES}")

    return avg_loss, overall_accuracy, correct, total_samples

def train(model, train_loader, val_loader, epochs, learning_rate, device, 
          val_frequency=1, weight_decay=0.0, lr_scheduler_step_size=7,
          global_model=None, mu=0.0, log_prefix="", logger=None):
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
    - global_model (nn.Module): The starting "global model" for the round.
                                Used to calculate the FedProx proximal term.
    - mu (float): The strength of the FedProx proximal term (the "elastic band").
    
    Returns:
    - model (nn.Module): The trained model.
    - history (list): A list of dictionaries containing metrics for each validation step.
    """
    
    # Fallback to root logger if specific logger not provided (Legacy compatibility)
    log = logger if logger else logging.getLogger()

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
        log.info(log_prefix + "Performing pre-training evaluation...")
        avg_val_loss, accuracy, correct, total = evaluate(model, val_loader, device, criterion)
        log.info(log_prefix + f"--- PRE-TRAINING VALIDATION ---")
        log.info(log_prefix + f"Validation Loss: {avg_val_loss:.4f} | "
                     f"Validation Acc: {accuracy:.2f}% ({correct}/{total})")
        history.append({
            'epoch': 0,
            'train_loss': None,
            'val_loss': avg_val_loss,
            'val_acc': accuracy
        })
    
    log.info(log_prefix + f"Starting training on {device} for {epochs} epochs...")

    if mu > 0 and global_model is not None:
        log.info(log_prefix + f"--- FedProx Training Enabled (mu={mu}) ---")
        # Ensure global model is on the same device
        global_model.to(device)
    
    for epoch in range(epochs):
        # --- 2. Training Phase ---
        model.train()
        running_loss = 0.0
        epoch_train_loss = 0.0
        train_batches = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for i, (data, labels) in enumerate(train_loader_tqdm):
            if (labels == -1).any():
                log.warning(log_prefix + "Skipping bad batch")
                continue
                
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Add FedProx "elastic band" term
            if mu > 0 and global_model is not None:
                proximal_term = 0.0
                # Iterate over trainable parameters only
                for (name, local_param), (global_name, global_param) in zip(
                    model.named_parameters(), global_model.named_parameters()
                ):
                    if local_param.requires_grad:
                        # Calculate the L2 norm (squared distance)
                        proximal_term += (local_param - global_param.to(device)).norm(2) ** 2
                
                loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            
            item_loss = loss.item()
            running_loss += item_loss
            epoch_train_loss += item_loss
            train_batches += 1
            
            if i % 10 == 9: 
                avg_loss = running_loss / 10
                train_loader_tqdm.set_postfix(loss=f"{avg_loss:.3f}")
                log.debug(log_prefix + f"Epoch {epoch+1}, Batch {i+1}, Avg. Loss: {avg_loss:.3f}")
                running_loss = 0.0
        
        # Step the learning rate scheduler
        scheduler.step()

        avg_epoch_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0.0
        log.info(log_prefix + f"Epoch {epoch+1}/{epochs} training complete. Avg Train Loss: {avg_epoch_train_loss:.4f}")

        # --- 3. Validation Phase ---

        val_loss, val_acc = None, None
        if val_loader and (epoch + 1) % val_frequency == 0:
            avg_val_loss, accuracy, correct, total = evaluate(model, val_loader, device, criterion)
            val_loss = avg_val_loss
            val_acc = accuracy
            
            log.info(log_prefix + f"--- Epoch {epoch+1}/{epochs} VALIDATION ---")
            log.info(log_prefix + f"Validation Loss: {avg_val_loss:.4f} | "
                          f"Validation Acc: {accuracy:.2f}% ({correct}/{total})")
            
        else:
            log.info(log_prefix + f"Epoch {epoch+1}/{epochs} training complete (validation skipped).")

        # Always push training metrics to history, even without validation
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_epoch_train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    log.info(log_prefix + "Finished Training")
    return model, history