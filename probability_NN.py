import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32, help='size of the batches')
parser.add_argument('--epo', type=int, default=3, help='size of epoches')
parser.add_argument('--dim', type=int, default=19, help='dim of data')
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
optimizer = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)
# optimizer = torch.optim.Adam(
#     network.parameters(), lr=0.0002, betas=(0.5, 0.999))

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if cuda:
    network.cuda()
    criterion.cuda()

# data = pd.read_csv('./dataDump/balance_data.csv').iloc[:, 1:]
data_df = pd.read_csv('./dataDump/vip_data.csv').iloc[:, 1:]
data_df = data_df.dropna()

# original data -> 0 for good guys, 1 for bad guys,so revert labels
# data_df.iloc[:, 0] = 1-data_df.iloc[:, 0]

labels = np.array(data_df.iloc[:, 0]).reshape(data_df.shape[0], -1)
data = np.array(data_df.iloc[:, 1:])
dataset_train, dataset_test, label_train, label_test = train_test_split(
    data, labels, test_size=0.2)

dataloader = DataLoader(
    dataset(dataset_train, label_train),
    batch_size=opt.bs,
    shuffle=True,
    drop_last=True
)

dataset_test = Variable(Tensor(dataset_test).type(Tensor))
label_test = Variable(Tensor(label_test).type(Tensor))

print('training...')
begin = datetime.now()

# Begin training
writer = SummaryWriter()
for epoch in range(opt.epo):
    for i, batch in enumerate(dataloader):

        # Configure input
        train = Variable(batch["data"].type(Tensor))
        label = Variable(batch["labels"].type(Tensor))

        network.train()
        optimizer.zero_grad()

        possibility = network(train)
        loss = criterion(possibility, label)

        loss.backward()
        optimizer.step()

        network.eval()
        # dataset_test
        pr = network(dataset_test)
        test_loss = criterion(pr, label_test)

        if i % 2 == 0:
            # accuracy
            predict_labels = torch.ge(pr, 0.5).float()
            correct = torch.eq(predict_labels, label_test).sum()
            acc = correct.item()/predict_labels.shape[0]

            # tensorboard visualize
            writer.add_scalars('Training&Testing Loss', {
                'Training Loss': loss.item(), 'Testing Loss': test_loss.item()}, i)
            writer.add_scalar('Accuracy', acc, i)

            # print log
            print(
                "[Epoch %d/%d] [Batch %d/%d] [trainLoss: %f] [testLoss: %f] [acc: %f]"
                % (epoch, opt.epo, i, len(dataloader), loss.item(), test_loss.item(), acc)
            )

writer.close()
end = datetime.now()
print("total time:", (end-begin).seconds)
