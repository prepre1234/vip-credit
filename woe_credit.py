import torch
from torch import nn
from torch.autograd import Variable

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import woe.feature_process as fp
import woe.eval as eval

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


data = pd.read_csv('./data/cs-training.csv').iloc[:, 1:]
data.rename(columns={'SeriousDlqin2yrs': 'target'}, inplace=True)
data = data.dropna()

''' woe分箱, iv and transform '''
print("woe....")
data_woe = data  # 用于存储所有数据的woe值
info_value_list = []
n_positive = sum(data['target'])
n_negtive = len(data) - n_positive
for column in list(data.columns[1:]):
    if data[column].dtypes == 'object':
        info_value = fp.proc_woe_discrete(
            data, column, n_positive, n_negtive, 0.05*len(data), alpha=0.05)
    else:
        info_value = fp.proc_woe_continuous(
            data, column, n_positive, n_negtive, 0.05*len(data), alpha=0.05)
    info_value_list.append(info_value)
    data_woe[column] = fp.woe_trans(data[column], info_value)

info_df = eval.eval_feature_detail(info_value_list, './dataDump/woe_info.csv')

# 删除iv值过小的变量
iv_threshold = 0.001
iv = info_df[['var_name', 'iv']].drop_duplicates()
x_columns = list(iv.var_name[iv.iv > iv_threshold])

data_woe = data_woe[x_columns]
data_woe.to_csv('./dataDump/data_woe.csv')


labels = np.array(data.iloc[:, 0]).reshape(data.shape[0], -1)
data_train = np.array(data_woe)

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


# -------------credit scores--------------

print('credit scores...')

W = np.array(LogRe.state_dict()['model.0.weight'])
b = np.array(LogRe.state_dict()['model.0.bias'])

ln_odds = np.dot(np.array(data_train), W.T)+b
credit = 487.122-28.8539*ln_odds
pd.DataFrame(credit).to_csv("credit_scores.csv")
