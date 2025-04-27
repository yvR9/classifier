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
def train(dataloader, model, loss_fn, optimizer, scheduler=None):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate if scheduler is provided
        if scheduler:
            scheduler.step()

        # Accumulate loss and correct predictions
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / size
    print(f"Train Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")


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

def main():
    # Load Fashion MNIST dataset
    DATA_DIR = "."
    train_dataset = datasets.FashionMNIST(DATA_DIR, train=True, transform=ToTensor(), download=True)
    test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, transform=ToTensor(), download=True)
    
    # Create data loaders
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = NeuralNetwork().to(device)
    print(model)
    
    # Define loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # Training loop
    epochs = 25
    best_accuracy = 0
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimiser)
        accuracy = test(test_dataloader, model, loss_fn)
        
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
        file_path = input("Please enter a filepath:\n> ")
        if file_path.lower() == "exit":
            print("Exiting...")
            break
        
        predicted_class = predict_image(model, file_path, device)
        print(f"Classifier: {predicted_class}")
 
if __name__ == "__main__":
    main()
