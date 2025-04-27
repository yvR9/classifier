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
      
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define testing function used to test model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

# Function to predict the class of a single image
def predict_image(model, image_path, device="cpu"):
    # Class labels for FashionMNIST
    class_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    try:
        # Load and preprocess the image
        img = io.read_image(image_path, mode=ImageReadMode.GRAY)
        img = img.float() / 255.0  # Normalize to [0, 1]
        
        # Check image dimensions
        if img.shape[1] != 28 or img.shape[2] != 28:
            img = transforms.Resize((28, 28))(img)
        
        # Add batch dimension
        img = img.unsqueeze(0).to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            predicted_class = class_labels[predicted.item()]
        
        return predicted_class
    
    except Exception as e:
        return f"Error processing image: {str(e)}"
