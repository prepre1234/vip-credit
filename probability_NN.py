import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32, help='size of the batches')
parser.add_argument('--epo', type=int, default=1, help='size of epoches')
parser.add_argument('--dim', type=int, default=10, help='dim of data')
opt = parser.parse_args()
print("args:")
print(opt)


class dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        if self.data.shape[0] != self.labels.shape[0]:
            raise Exception(
                'number of the data should be equal to number of labels!')

    def __getitem__(self, index):
        data = self.data[index % self.data.shape[0]]
        label = self.labels[index % self.data.shape[0]]

        return {"data": data, "labels": label}

    def __len__(self):
        return self.data.shape[0]


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.dim, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 64),
            *block(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Loss function
criterion = nn.BCELoss()

# Initialize network
network = Network()

# Optimizer
optimizer = torch.optim.Adam(
    network.parameters(), lr=0.0002, betas=(0.5, 0.999))

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if cuda:
    network.cuda()
    criterion.cuda()

data = pd.read_csv('./dataDump/balance_data.csv').iloc[:, 1:]
# original data -> 0 for good guys, 1 for bad guys,so revert labels
labels = 1-np.array(data.iloc[:, 0]).reshape(data.shape[0], -1)
data_train = np.array(data.iloc[:, 1:])

dataloader = DataLoader(
    dataset(data_train, labels),
    batch_size=opt.bs,
    shuffle=True,
    drop_last=True
)


print('training...')
begin = datetime.now()

# Begin training
for epoch in range(opt.epo):
    for i, batch in enumerate(dataloader):

        # Configure input
        train = Variable(batch["data"].type(Tensor))
        label = Variable(batch["labels"].type(Tensor))

        optimizer.zero_grad()

        possibility = network(train)
        loss = criterion(possibility, label)

        loss.backward()
        optimizer.step()

        # print log
        print(
            "[Epoch %d/%d] [Batch %d/%d] [Loss: %f]"
            % (epoch, opt.epo, i, len(dataloader), loss.item())
        )

end = datetime.now()
print("total time:", (end-begin).seconds)


# -------------save probability--------------
'''test with train dataset'''

print('save probablility...')

network.eval()
probability = pd.DataFrame(
    network(Variable(Tensor(data_train).type(Tensor))).detach().numpy())
probability.to_csv('./dataDump/probability_NN.csv')
