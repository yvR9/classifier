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
