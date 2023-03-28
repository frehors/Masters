import pandas as pd
# load data into train, test and validation sets using sklearn then to tensor for pytorch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch

# Prepare data in one place
data = pd.read_csv('data/avocado.csv')
data = pd.get_dummies(data, columns=['type', 'region'], drop_first=True)
# make date into year month and day, maybe dummies is better
data['year'] = pd.to_datetime(data['Date']).dt.year
data['month'] = pd.to_datetime(data['Date']).dt.month
data['day'] = pd.to_datetime(data['Date']).dt.day
# sort by date ascending
data.sort_values(by='Date', inplace=True)
data.drop('Date', axis=1, inplace=True)


# split data into train, validation, and test sets
#train, test = train_test_split(data, test_size=0.2, random_state=42)
#train, val = train_test_split(train, test_size=0.2, random_state=42)
# make 80% train, 10% validation, 10% test using indeces
train = data.iloc[:int(len(data)*0.8)]
val = data.iloc[int(len(data)*0.8):int(len(data)*0.9)]
test = data.iloc[int(len(data)*0.9):]
y_train = train['AveragePrice']
y_val = val['AveragePrice']
y_test = test['AveragePrice']
train.drop('AveragePrice', axis=1, inplace=True)
val.drop('AveragePrice', axis=1, inplace=True)
test.drop('AveragePrice', axis=1, inplace=True)


# make categorical variables into dummy variables

# create dataloaders for train, validation, and test sets, predict average price, using pytorch dataloaders
class Dataset(Dataset):
    def __init__(self, data, labels):
        # make data and labels into tensors
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def createLoaders(batch_size, shuffle, num_workers):
    train_dataset = Dataset(train, y_train)
    val_dataset = Dataset(val, y_val)
    test_dataset = Dataset(test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, val_loader, test_loader










