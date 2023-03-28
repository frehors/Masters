import torch
import torch.nn as nn
import torch.nn.functional as F
from data import Dataset, data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# create network class with training and validation methods
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        self.device = device
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.losses = {'train': np.zeros(1), 'val': np.zeros(1)}

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def train(self, train_loader, val_loader, epochs, learning_rate):
        # define loss function and optimizer
        self.losses = {'train': np.zeros(epochs), 'val': np.zeros(epochs)}
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # create lists to store losses
        train_losses = []
        val_losses = []
        # train network
        for epoch in range(epochs):
            for batch in train_loader:

                # get data and labels to device
                batch_data, batch_labels = batch
                batch_data.to(self.device)
                batch_labels.to(self.device)
                # set gradients to zero
                optimizer.zero_grad()
                # forward pass
                output = self(batch_data)
                # calculate loss
                loss = loss_fn(output, batch_labels)
                # backward pass
                loss.backward()
                # update weights
                optimizer.step()
            # calculate losses

            train_loss = self.evaluate(train_loader, loss_fn)
            val_loss = self.evaluate(val_loader, loss_fn)
            self.losses['train'][epoch] = train_loss
            self.losses['val'][epoch] = val_loss
            # print losses
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # plot losses
        #plt.plot(train_losses, label='Train Loss')
        #plt.plot(val_losses, label='Val Loss')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.show()

    def evaluate(self, loader, loss_fn):
        # get loss using loader and loss function
        loss = 0
        for batch in loader:
            batch_data, batch_labels = batch
            batch_data.to(self.device)
            batch_labels.to(self.device)
            output = self(batch_data)
            loss += loss_fn(output, batch_labels).item()
        return loss / len(loader)

    def plot_loss(self, log=False):
        # plot losses
        if log:
            plt.plot(np.log(self.losses['train']), label='Train Loss')
            plt.plot(np.log(self.losses['val']), label='Val Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.yscale('log')
            plt.legend()
            plt.show()
        else:
            plt.plot(self.losses['train'], label='Train Loss')
            plt.plot(self.losses['val'], label='Val Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

