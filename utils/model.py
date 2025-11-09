import logging
import pickle
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

NUM_CLASSES = 40


class Model(nn.Module):
    """
    This class wraps the torch model
    More fields can be added here

    """

    def __init__(self):
        """
        Constructor

        """
        super().__init__()
        self.model_change = None
        self._param_count_ot = None
        self._param_count_total = None
        self.accumulated_changes = None
        self.shared_parameters_counter = None

    def count_params(self, only_trainable=False):
        """
        Counts the total number of params

        Parameters
        ----------
        only_trainable : bool
            Counts only parameters with gradients when True

        Returns
        -------
        int
            Total number of parameters

        """
        if only_trainable:
            if not self._param_count_ot:
                self._param_count_ot = sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                )
            return self._param_count_ot
        else:
            if not self._param_count_total:
                self._param_count_total = sum(p.numel() for p in self.parameters())
            return self._param_count_total

    def rewind_accumulation(self, indices):
        """
        resets accumulated_changes at the given indices

        Parameters
        ----------
        indices : torch.Tensor
            Tensor that contains indices corresponding to the flatten model

        """
        if self.accumulated_changes is not None:
            self.accumulated_changes[indices] = 0.0

    def dump_weights(self, directory, uid, round):
        """
        saves the current model as a pt file into the specified direcectory

        Parameters
        ----------
        directory : str
            directory in which the weights are dumped
        uid : int
            uid of the node, will be used to give the weight a unique name
        round : int
            current round, will be used to give the weight a unique name

        """
        torch.save(self.state_dict(), Path(directory) / f"{round}_weight_{uid}.pt")

    def get_weights(self):
        """
        flattens the current weights

        """
        with torch.no_grad():
            tensors_to_cat = []
            for _, v in self.state_dict().items():
                tensors_to_cat.append(v.flatten())
            flat = torch.cat(tensors_to_cat)

        return flat
    

class FoodClassifier(Model):
    """
    A CPU-friendly transfer learning model using MobileNetV3-Small.
    This model freezes the pre-trained feature extractor and only
    trains a new final classifier layer, making it fast on CPU.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Load a pre-trained MobileNetV3-Small
        # We use the "DEFAULT" weights, which are the best available
        logging.info("Loading pre-trained MobileNetV3-Small weights...")
        self.base_model = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        )
        
        # 2. Freeze all feature extraction layers
        # This is to makr it fast on a CPU
        logging.info("Freezing base model parameters...")
        for param in self.base_model.parameters():
            param.requires_grad = False

        # MobileNetV3-Small has 13 'features'. We unfreeze the last 2.
        logging.info("Fine-Tuning: Unfreezing last 2 feature blocks (11 and 12)...")
        for param in self.base_model.features[11:].parameters():
            param.requires_grad = True
            
        # 3. Replace the final classifier layer
        # Get the number of input features for the original classifier
        in_features = self.base_model.classifier[0].in_features  # 576
        hidden_features = self.base_model.classifier[0].out_features # 1024

        logging.info(f"Rebuilding classifier with {in_features} -> {hidden_features} -> {NUM_CLASSES} and p=0.4 dropout.")
        
        # Create a new, untrained linear layer for our 40 classes
        # Only this layer's parameters will have requires_grad = True
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Hardswish(),
            nn.Dropout(p=0.4, inplace=True), # Increased from 0.2 to 0.4
            nn.Linear(hidden_features, NUM_CLASSES)
        )
        
        # Reset parameter counts so they are recalculated
        self._param_count_ot = None
        self._param_count_total = None

    def forward(self, x):
        """
        Forward pass of the model
        """
        return self.base_model(x)



# NOT USED FOR THIS HACKATHON
class LeNet(Model):
    """
    Class for a LeNet-inspired Model adapted for 224x224x3 inputs.
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791

    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 10 output classes

        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.gn1 = nn.GroupNorm(2, 32)
        self.conv2 = nn.Conv2d(32, 32, 5, padding="same")
        self.gn2 = nn.GroupNorm(2, 32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding="same")
        self.gn3 = nn.GroupNorm(2, 64)
        self.fc1 = nn.Linear(64 * 28 * 28, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            The input torch tensor

        Returns
        -------
        torch.tensor
            The output torch tensor

        """
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc1(x)
        return x