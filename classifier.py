import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, io, transforms
from torchvision.io import ImageReadMode
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# Checks for available device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Define training function used to train model
def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
       