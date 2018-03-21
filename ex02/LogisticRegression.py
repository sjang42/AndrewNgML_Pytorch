import math
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable

# 데이터 얻기
# 모델 정의
# 로스와 옵티마이저선택
# 학습

data = pd.read_csv('data/ex2data1.txt', header=None)

X = data.iloc[:, :2].as_matrix()
Y = data.iloc[:, 2].as_matrix()
print(type(Y))


def plot_data1(X, Y):
    pos = np.where(Y == 1)
    neg = np.where(Y == 0)

    plt.figure()
    plt.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'yo', markersize=7)
    plt.show()


class ExamDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        :param csv_file (stirng): Path to the csv file.
        :param transform (Callable, optional): Optional transform to be applied to a sample.
        """

        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.X = self.data.iloc[:, :2].as_matrix()
        self.Y = self.data.iloc[:, 2].as_matrix()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'X': X[idx], 'Y': Y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        X, Y = sample['X'].astype(float), sample['Y']

        return {'X': torch.from_numpy(X).float(), 'Y': torch.LongTensor([int(Y)])}


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_class):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_class)

    def forward(self, x):
        out = self.linear(x)
        return out


input_size = 2
num_class = 2
total_epoch = 500
learning_rate = 0.00001
batch_size = 100

model = LogisticRegression(input_size, num_class)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
exam_dataset = ExamDataset('data/ex2data1.txt', ToTensor())
dataloader = DataLoader(dataset=exam_dataset, batch_size=batch_size, num_workers=4)

for epoch in range(total_epoch):
    for idx, sample_batched in enumerate(dataloader):
        inputs = Variable(sample_batched['X'])
        labels = Variable(sample_batched['Y'])
        labels = torch.squeeze(labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 4 == 0:
        print('Epoch: [%d/%d], Loss: %.4f' %(epoch+1, total_epoch, loss.data[0]))


def sigmoid(z):
    return 1 / (1 + math.e ** -z)


test_in = Variable(torch.Tensor([[20, 20], [50, 50], [100, 100], [60.18259938620976, 86.30855209546826]]))
out = model(test_in)

print(out)
print(sigmoid(out.data.numpy()))
