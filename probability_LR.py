import torch
from torch import nn
from torch.autograd import Variable

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=10, help='dim of data')
parser.add_argument('--epo', type=int, default=10000, help='size of eopches')
opt = parser.parse_args()
print('args:')
print(opt)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Loss function
criterion = nn.BCELoss()

# Initialize LR model
LogRe = LogisticRegression()

# Optimizer
optimizer = torch.optim.Adam(
    LogRe.parameters(), lr=0.0002, betas=(0.5, 0.999))

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if cuda:
    LogRe.cuda()
    criterion.cuda()

data = pd.read_csv('./dataDump/balance_data.csv').iloc[:, 1:]
# original data -> 0 for good guys, 1 for bad guys,so revert labels
labels = 1-np.array(data.iloc[:, 0]).reshape(data.shape[0], -1)
data_train = np.array(data.iloc[:, 1:])

# Configure input
data_train = Variable(Tensor(data_train).type(Tensor))
labels = Variable(Tensor(labels).type(Tensor))

print("training...")
begin = datetime.now()

# Begin training
for epoch in range(opt.epo):

    optimizer.zero_grad()

    out = LogRe(data_train)
    loss = criterion(out, labels)

    loss.backward()
    optimizer.step()

    # print log
    print(
        "[Epoch %d/%d] [Loss: %f]"
        % (epoch, opt.epo, loss.item())
    )

end = datetime.now()
print("total time:", (end-begin).seconds)


# -------------save probability--------------
'''test with train dataset'''

print('save probablility...')

LogRe.eval()
probability = pd.DataFrame(
    LogRe(Variable(Tensor(data_train).type(Tensor))).detach().numpy())
probability.to_csv('./dataDump/probability_LR.csv')
