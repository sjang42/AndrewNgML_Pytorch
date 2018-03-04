import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


# define linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


# visualize loss history
def plotLossHistory(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.show()


# load data as numpy array
xy = pd.read_csv('data/ex1data2.txt', header=None)
x_data = xy.iloc[:, :-1].as_matrix()
y_data = xy.iloc[:, -1].as_matrix()


# normalize x_data
mean = np.mean(x_data, 0)
std = np.std(x_data, 0)
x_data = (x_data - mean) / std

# convert numpy array to torch Variable
x_data = Variable(torch.from_numpy(x_data).float())
y_data = Variable(torch.from_numpy(y_data).float())

# define hyper parameter
total_epoch = 400
learning_rate = 0.01

# define loss and optimizer
input_size = 2
output_size = 1
model = LinearRegression(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_history = []


# train
for epoch in range(total_epoch):
    optimizer.zero_grad()
    outputs = model(x_data)
    targets = y_data

    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()
    loss_history.append(loss.data[0])
    if (epoch + 1) % 5 is 0:
        print('Epoch: [%d/%d], Loss: %e' % (epoch+1, total_epoch, loss.data[0]))

# visualize loss history
plotLossHistory(loss_history)
