import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image, ImageReadMode
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
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct  # Return accuracy for model saving logic

# Function to predict the class of a single image
def predict_image(model, image_path, device="cpu"):
    # Class labels for FashionMNIST
    class_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    try:
        # Load and preprocess the image
        
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

def main():
    # Load Fashion MNIST dataset
    DATA_DIR = "."
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.FashionMNIST(DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, transform=transform, download=True)
    
    # Create data loaders
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = NeuralNetwork().to(device)
    print(model)
    
    # Define loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    
    
    # Training loop
    epochs = 25
    best_accuracy = 0
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved new best model with accuracy: {(100*best_accuracy):>0.1f}%")
    
    print("Training complete!")
    
    # Load best model for evaluation
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    print("Loaded best model for evaluation")
    
    while True:
        
        if file_path.lower() == "exit":
            print("Exiting...")
            break
        
        predicted_class = predict_image(model, file_path, device)
        print(f"Classifier: {predicted_class}")
 
if __name__ == "__main__":
    main()