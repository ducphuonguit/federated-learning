import flwr as fl
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from model import Net

# Load MNIST dataset
def load_data():
    # Downloads the MNIST dataset and applies a transformation to convert images to tensors (ToTensor() normalizes pixel values to [0, 1]).
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    # 80% for training (trainset)
    train_len = int(0.8 * len(dataset))
    # 20% for validation (valset)
    val_len = len(dataset) - train_len
    trainset, valset = random_split(dataset, [train_len, val_len])
    return trainset, valset

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        # Net is the local PyTorch neural network.
        self.model = Net()
        # trainset and valset hold the local training and validation data.
        self.trainset, self.valset = load_data()
        # CrossEntropyLoss is used for multi-class classification.
        self.criterion = nn.CrossEntropyLoss()
        # Adam optimizer with a learning rate of 0.01.
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        print("Sending model parameters to server...")
        # Converts the model parameters (weights and biases) into a list of NumPy arrays, These parameters are sent to the Flower server when requested.
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # The set_parameters method receives the model parameters from the server and updates the local model with these parameters.    
    def set_parameters(self, parameters):
        print("Receiving model parameters from server...", parameters)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

# Input: parameters (global model sent from the server).
# Process:
# Updates the local model using set_parameters.
# Trains for 1 epoch on the local training data using a DataLoader (batch size = 32).
# Loss is computed using CrossEntropyLoss.
# Parameters are updated using backpropagation (loss.backward() and optimizer.step()).
# Output:
# Returns updated model parameters.
# Returns the number of training samples (len(self.trainset)).
    def fit(self, parameters, config):
        print("Training model locally...")
        self.set_parameters(parameters)
        trainloader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        self.model.train()
        for epoch in range(1):  # One epoch
            for images, labels in trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters({}), len(self.trainset), {}

# Input: parameters (global model sent from the server).
# Process:
# Updates the local model using set_parameters.
# Evaluates the model on the local validation set (no gradient computation using torch.no_grad()).
# Computes loss and accuracy.
# Output:
# Returns the loss value.
# Number of validation samples.
# Dictionary containing the accuracy.
    def evaluate(self, parameters, config):
        print("Evaluating model locally...")
        self.set_parameters(parameters)
        valloader = DataLoader(self.valset, batch_size=32)
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(self.valset)
        print(f"Validation Results: Loss = {loss:.4f}, Accuracy = {accuracy:.4%}")
        return loss, len(self.valset), {"accuracy": accuracy}

# Start the client
# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
