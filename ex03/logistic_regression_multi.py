import scipy.io as sio
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.utils import shuffle

# get data and shuffle
matdata = sio.loadmat('data/ex3data1.mat')
X = matdata['X']
y = matdata['y']
X, y = shuffle(X, y, random_state=0)

num_trainset = 4500
X_train = X[:num_trainset, :]
y_train = y[:num_trainset, :]


X_test = X[num_trainset:, :]
y_test = y[num_trainset:, :]

y_train[y_train == 10] = 0
y_test[y_test == 10] = 0


# define model
class LogisticRegressionMulti(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionMulti, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


# set hyperparameter
input_size = 400
output_size = 10
model = LogisticRegressionMulti(input_size, output_size)

learning_rate = 0.1
total_epoch = 50
batch_size = 10


# set criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# train model
train_size = len(X_train)
for epoch in range(total_epoch):
    for iter in range(int(train_size / batch_size)):
        inputs = X_train[iter * batch_size:(iter+1) * batch_size, :]
        inputs = torch.from_numpy(inputs).float()
        inputs = Variable(inputs)

        labels = y_train[iter * batch_size:(iter+1) * batch_size, :]
        labels = torch.from_numpy(labels).long()
        labels = Variable(labels)
        labels = torch.squeeze(labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch: [%d/%d], Loss: %.4f' %(epoch+1, total_epoch, loss.data[0]))


# test model
correct = 0
test_size = 100
for idx in range(test_size):
    images = Variable(torch.from_numpy(X_test[idx]).float())
    output = model(images)
    target = y_test[idx]
    predicted = np.argmax(output.data)
    correct += predicted == target

precision = correct / test_size
print('precision = {precision}'.format(precision=precision))
