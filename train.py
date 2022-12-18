import torch
import torch.nn as nn
import torch.nn.functional as F
from feedForwardsNN import FeedForwardNN
from data import Dataset, data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
shuffle = True
num_workers = 0

# split data into train, validation, and test sets
train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# create dataset objects
train_dataset = Dataset(train, train['AveragePrice'])
val_dataset = Dataset(val, val['AveragePrice'])
test_dataset = Dataset(test, test['AveragePrice'])

# create data loader objects
trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
valLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# create model
inputSize = 12
hiddenSize = 24
outputSize = 1
epochs = 100
learningRate = 0.001


model = FeedForwardNN(inputSize, hiddenSize, outputSize)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# train model
trainLoss = []
valLoss = []
for epoch in range(epochs):
    # train model
    for data, target in trainLoader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    trainLoss.append(loss.item())
    # get val loss after training, every epoch
    for data, target in valLoader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
    valLoss.append(loss.item())
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss.item()))

# plot loss
plt.plot(trainLoss, label='Training loss')
plt.plot(valLoss, label='Validation loss')
plt.legend(frameon=False)
plt.show()

