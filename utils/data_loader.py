import os
import torch
from torch.utils.data import Dataset, DataLoader
import logging

class PreprocessedDataset(Dataset):
    """
    A custom Dataset class to load preprocessed .pt tensor files.
    Assumes a directory structure like:
    root_dir/
        0/
            img1.pt
            img2.pt
        1/
            img3.pt
        ...
        39/
            imgN.pt
    """
    def __init__(self, root_dir):
        self.samples = []
        if not os.path.exists(root_dir):
            logging.warning(f"Data directory not found: {root_dir}")
            return
            
        for label_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, label_dir)
            if not os.path.isdir(class_path):
                continue
            
            # Get label (class index) from directory name
            try:
                label = int(label_dir)
            except ValueError:
                # Skip directories that aren't class indices (e.g., '.ipynb_checkpoints')
                continue
                
            for file_name in os.listdir(class_path):
                if file_name.endswith('.pt'):
                    file_path = os.path.join(class_path, file_name)
                    self.samples.append((file_path, label))
                    
        if not self.samples:
            logging.warning(f"No .pt files found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load the preprocessed tensor
        # We handle potential load errors gracefully
        try:
            data = torch.load(file_path, weights_only=True)
            return data, label
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            # Return a dummy tensor and a placeholder label (e.g., -1)
            # Skipping complicates batching, but don't expect this error.
            return torch.zeros((3, 224, 224)), -1


def create_dataloader(data_dir, batch_size, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the PreprocessedDataset.
    
    Parameters:
    - data_dir (str): Path to the root directory of the dataset (e.g., .../client_0).
    - batch_size (int): How many samples per batch to load.
    - shuffle (bool): Whether to shuffle the data (True for train, False for val/test).
    - num_workers (int): How many subprocesses to use for data loading.
    
    Returns:
    - DataLoader: A PyTorch DataLoader instance.
    """
    dataset = PreprocessedDataset(root_dir=data_dir)
    
    if len(dataset) == 0:
        logging.warning(f"Cannot create DataLoader for empty dataset at {data_dir}")
        return None
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle  # Drop last incomplete batch if shuffling (training)
    )