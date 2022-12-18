import torch
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


data = pd.read_csv('data/avocado.csv')
dataset = Dataset(data, data['AveragePrice'])
