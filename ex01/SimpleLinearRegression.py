import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

data = pd.read_csv('data/ex1data1.txt', header=None)
X = data.iloc[:, 0].as_matrix()
X = np.asmatrix(X)
X = np.reshape(X, (-1, 1))

Y = data.iloc[:, 1].as_matrix()
Y = np.asmatrix(Y)
Y = np.reshape(Y, (-1, 1))


def plot_data(X, Y):
    plt.figure()
    plt.grid()
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.plot(X, Y, 'bo', markersize=3)
    plt.show()


plot_data(X, Y)


class ProfitDataset(Dataset):
    """Puopulation dataset"""
    def __init__(self, csv_file, transform=None):
        """
        Args:
        :param csv_file (string): Path to the csv file.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        population = self.data.iloc[idx, 0]
        profit = self.data.iloc[idx, 1]
        sample = {'population': population, 'profit':  profit}

        if self.transform:
            sample = self.transform(sample)

        return sample


profit_dataset = ProfitDataset(csv_file='data/ex1data1.txt')

for i in range(len(profit_dataset)):
    sample = profit_dataset[i]
    print(i, sample['population'], sample['profit'])

    if i is 4:
        break


class ToTensor(object):
    """Convert ndarray in sample to Tensors"""

    def __call__(self, sample):
        population, profit = sample['population'], sample['profit']

        return {'population': torch.Tensor([population]),
                'profit': torch.Tensor([profit])}


transformed_dataset = ProfitDataset(csv_file='data/ex1data1.txt', transform=ToTensor())

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['population'].size(), sample['profit'].size())
    if i is 3:
        break

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)


def plot_profit_batch(sample_batched):
    population_batch, profit_batch =\
            sample_batched['population'], sample_batched['profit']
    # batch_size=len(population_batch)
    plot_data(population_batch.numpy(), profit_batch.numpy())


class SimpleLinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


input_size = 1
output_size = 1
num_epoch = 1000
learning_rate = 0.001

model = SimpleLinearRegression(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
dataloader = DataLoader(transformed_dataset, batch_size=97, shuffle=True, num_workers=4)

for epoch in range(num_epoch):
    for idx, sample_batched in enumerate(dataloader):
        inputs = Variable(sample_batched['population'])
        targets = Variable(sample_batched['population'])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 4 == 0:
        print('Epoch: [%d/%d], Loss: %.4f' %(epoch+1, num_epoch, loss.data[0]))


predicted = model(Variable(torch.from_numpy(X).float())).data.numpy()
plt.plot(X, Y, 'bo', label='Original data')
plt.plot(X, predicted, label='Fitted line')
plt.legend()
plt.show()
