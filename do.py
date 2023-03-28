from data import createLoaders
from network import FeedForwardNN
import torch
# create dataloaders
train_loader, val_loader, test_loader = createLoaders(batch_size=64, shuffle=False, num_workers=0)

# create network
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('running on', device)
input_size = train_loader.dataset.data.shape[1]
model = FeedForwardNN(input_size, 50, 1, device)
lr = 0.001
epochs = 150
model.train(train_loader, val_loader, epochs, lr)
model.plot_loss(log=True)
