import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(2809)


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_class):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_class)

    def initialize(self):
        nn.init.xavier_uniform(self.linear.weight.data)
        print('hi')
        self.linear.bias.data.zero_()

    def forward(self, x):
        out = self.linear(x)
        return F.sigmoid(out)


data = pd.read_csv('data/ex2data1.txt', header=None)

X = Variable(torch.from_numpy(data.iloc[:, :2].as_matrix()).float())
Y = Variable(np.reshape(torch.from_numpy(data.iloc[:, 2].as_matrix()).float(),
                        (-1, 1)))
input_size = 2
num_class = 1
total_epoch = 250000
learning_rate = 0.0001

model = LogisticRegression(input_size, num_class)
criterion = nn.BCELoss()
optimizer = torch(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


loss_history = []
for epoch in range(total_epoch):
    output = model(X)

    optimizer.zero_grad()
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch: [%d/%d], Loss: %.4f' %(epoch+1, total_epoch, loss.data[0]))
    loss_history.append(loss.data[0])

input = Variable(torch.Tensor([20, 30]))
print("predict", (20, 30), model(input).data)

input = Variable(torch.Tensor([80, 30]))
print("predict", (80, 30), model(input).data)

input = Variable(torch.Tensor([10, 10]))
print("predict", (10, 10), model(input).data)

input = Variable(torch.Tensor([45, 85]))
print("predict", (45, 85), model(input).data)


def plotLossHistory(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.show()

plotLossHistory(loss_history)
