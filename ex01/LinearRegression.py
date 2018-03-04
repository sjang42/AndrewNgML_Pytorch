from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch
from torchvision import transforms
from torch.autograd import Variable


# datase of housing dataset from coursera
class HousePriceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        :param csv_file (stirng): Path to the csv file.
        :param transform (Callable, optional): Optional transform to be applied to a sample.
        """

        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data.iloc[idx, :2]
        Y = self.data.iloc[idx, 2]
        sample = {'X': X, 'Y': Y}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Normalize with normal distribution
class ToNormalDistribution(object):
    def __init__(self, mean, std):
        """
        :param mean (int or array like): mean of X
        :param std (float or array like): std of X
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        x = sample['X']
        x_normal = (x - self.mean) / self.std
        sample['X'] = x_normal
        return sample


# convert numpy array to tensor
class ToTensor(object):
    def __call__(self, sample):
        X, Y = sample['X'].as_matrix(), np.float(sample['Y'])
        X = X.astype(float)
        return {'X': torch.from_numpy(X).float(), 'Y': torch.Tensor([Y])}


# plot 3d to visualize housing price dataset
def plot_3d(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, linewidth=0.02, antialiased=True)

    plt.show()


# Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


# calculate mean and std from dataset
price_dataset = HousePriceDataset(csv_file='data/ex1data2.txt')
mean = np.mean(price_dataset.data.iloc[:, :2].as_matrix(), 0)
std = np.std(price_dataset.data.iloc[:, :2].as_matrix(), 0)

# Housing price dataset
price_dataset = HousePriceDataset(csv_file='data/ex1data2.txt',
                                  transform=transforms.Compose([
                                      ToNormalDistribution(mean, std),
                                      ToTensor()
                                  ]))

input_size = 2
output_size = 1

# set hyper paramerters
num_epoch = 400
learning_rate = 0.01

# batch size is same with all data.
# because data is so small, we don't need to batch.
batch_size = len(price_dataset)

# set criterion and optimizer after model
model = LinearRegression(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
dataloader = DataLoader(price_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
loss_history = []


# train
for epoch in range(num_epoch):
    for idx, sample_batched in enumerate(dataloader):
        inputs = Variable(sample_batched['X'])
        targets = Variable(sample_batched['Y'])
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        # save loss history for visualizing
        loss_history.append(loss.data[0])
    if (epoch + 1) % 5 is 0:
        print('Epoch: [%d/%d], Loss: %e' % (epoch+1, num_epoch, loss.data[0]))


# visualize loss history
def plotLossHistory(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.show()

plotLossHistory(loss_history)
