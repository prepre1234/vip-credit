from utils import computeTime
from utils import readDataFromMysql
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

# parser = argparse.ArgumentParser()
# parser.add_argument('--bs', type=int, default=32, help='size of the batches')
# parser.add_argument('--epo', type=int, default=10, help='size of epoches')
# parser.add_argument('--dim', type=int, default=19, help='dim of data')
# opt = parser.parse_args()
# print("args:")
# print(opt)

bs = 32
epo = 50
decay_start_epoch = 5
lr = 1e-3
beta1 = 0.9
beta2 = 0.9


pub = False
zhiding = False
tensorboard = False

# ds:data_scale
if not pub:
    dim = 19
    ds = 1
    # ts:test_size
    ts = 0.01
else:
    dim = 10
    if zhiding:
        tsize = 350
        vsize = 2400
        ds = (tsize+vsize)/120269
        ts = vsize/(tsize+vsize)
    else:
        ds = 1
        ts = 0.2

if not pub:
    ##############################
    sql = 'select zjc_supplier.name,zjc_supplier.province,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average,zjc_supplier.regchecktime,zjc_supplier.createtime from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2;'
    # pre_data:将要预测的供应商（所有供应商，包括训练用的正负样本）
    pre_data = readDataFromMysql(sql)
    # name_region：前两列是供应商的名字和地区，保存留到后面使用
    name_region = pre_data[[0, 1]]
    # pre_data:分离name_region后的剩余列数据
    pre_data = pre_data.iloc[:, 2:]
    total = pre_data.shape[0]

    # 最后两列是注册和审核时间，用于计算已经成为平台用户的时长
    time = computeTime(pre_data.iloc[:, 18:])
    # 将注册和审核时间替换为时长
    pre_data = pd.concat([pre_data.iloc[:, 0:18], time], axis=1)
    pre_data = pre_data.astype('float32')
    pre_data = np.array(pre_data)

    # 标准化处理
    scaler = preprocessing.StandardScaler().fit(pre_data)
    pre_data = scaler.transform(pre_data)

    ##############################


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


class ResidualBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim)
        )

    def forward(self, x):
        return x+self.block(x)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        model = [
            *block(dim, 64),
            ResidualBlock(64),
            *block(64, 128),
            ResidualBlock(128),
            *block(128, 256),
            ResidualBlock(256),
            *block(256, 64),
            ResidualBlock(64),
            *block(64, 1),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

    # if isinstance(m, nn.Linear):
        # torch.nn.init.kaiming_normal_(m)


# Loss function
criterion = nn.BCELoss()

# Initialize network
network = Network()
network.apply(weights_init)

# Optimizer
# optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adam(
    network.parameters(), lr=lr, betas=(beta1, beta2))

# lr_decay
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: 1-max(0, epoch-decay_start_epoch)/(epo-decay_start_epoch))

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if cuda:
    network.cuda()
    criterion.cuda()

if pub:
    data_df = pd.read_csv('./data/cs-training.csv').iloc[:, 1:]
else:
    # data_df：目前的VIP正负样本
    data_df = pd.read_csv('./dataDump/vip_data.csv').iloc[:, 1:]
    # data_df = data_df.iloc[:, [0, 3, 8, 10, 11, 12, 13, 14, 15, 16, 19]]

data_df = data_df.dropna()

# 实验时按比例ds取部分数据集作为训练集+测试集，如果用全部数据，则ds=1
data_df = data_df.iloc[0:int(data_df.shape[0]*ds)]

# data_df[pd.DataFrame(data_df.iloc[:, 0]).duplicated()] #查找正负样本重叠部分（数据给的有问题）
name_set = set(data_df.iloc[:, 0])
labels = np.array(data_df.iloc[:, 1]).reshape(data_df.shape[0], -1)
data = np.array(data_df.iloc[:, 2:])

if pub:
    scaler = preprocessing.StandardScaler().fit(data)

data = scaler.transform(data)


dataset_train, dataset_test, label_train, label_test = train_test_split(
    data, labels, test_size=ts)

data_size = data_df.shape[0]
train_size = dataset_train.shape[0]
test_size = dataset_test.shape[0]

dataloader = DataLoader(
    dataset(dataset_train, label_train),
    batch_size=bs,
    shuffle=True,
    drop_last=True
)

dataset_test = Variable(Tensor(dataset_test).type(Tensor))
label_test = Variable(Tensor(label_test).type(Tensor))

print('training...')
begin = datetime.now()

# Begin training
writer = SummaryWriter()
for epoch in range(epo):
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
            if tensorboard:
                writer.add_scalars('Training&Testing Loss', {
                    'Training Loss': loss.item(), 'Testing Loss': test_loss.item()}, i)
                writer.add_scalar('Accuracy', acc, i)

            # print log
            print(
                "[size:%d/%d/%d] [Epoch %d/%d] [Batch %d/%d\t] [trainLoss: %f] [testLoss: %f] [acc: %f] [lr: %f]"
                % (data_size, train_size, test_size, epoch, epo, i, len(dataloader), loss.item(), test_loss.item(), acc, scheduler.get_lr()[0])
            )

    scheduler.step()

writer.close()
end = datetime.now()
print("total time:", (end-begin).seconds)

if not pub:
    # prediction
    network.eval()
    pre_data = Variable(Tensor(pre_data).type(Tensor))
    # pre_result预测的概率
    pre_result = network(pre_data)

    will = torch.ge(pre_result, 0.5).float().sum()
    no = torch.lt(pre_result, 0.4).float().sum()

    # 会充值/0.4-0.5二次充值/不会充值
    print('>=0.5:\t', will)
    print('>=0.4&&<0.5\t:', total-will-no)
    print('<0.4:\t', no)

    pre_result = pd.DataFrame(pre_result.detach().numpy())
    # pre_result更新为：名字、地区、概率
    pre_result = pd.concat([name_region, pre_result], axis=1)
    # pre_positive预测会充值的供应商
    pre_positive = pre_result[pre_result.iloc[:, 2] >= 0.5]
    pre_positive.columns = ['name', 'province', 'prob']

    # 去除正负样本中的供应商
    tmp = []
    for row in pre_positive.itertuples():
        name = getattr(row, 'name')
        if name not in name_set:
            tmp.append(row)

    pre_positive = pd.DataFrame(tmp)
    pre_positive.sort_values('prob', ascending=False, inplace=True,)
    print('最终预测VIP充值的有：', len(pre_positive))

    Yangtze_River_Delta = set({'上海市', '江苏省', '浙江省', '安徽省'})
    # top50/长三角/其他
    top = pre_positive.iloc[0:50]
    top_Yangtze_River_Delta = []
    top_other = []
    for row in top.itertuples():
        province = getattr(row, 'province')
        if province in Yangtze_River_Delta:
            top_Yangtze_River_Delta.append(row)
        else:
            top_other.append(row)
    top_Yangtze_River_Delta = pd.DataFrame(top_Yangtze_River_Delta)
    top_other = pd.DataFrame(top_other)
    print('Top50 长三角地区：', len(top_Yangtze_River_Delta))
    print('Top50 其余地区：', len(top_other))

    # remain/长三角/其他
    remain = pre_positive.iloc[50:]
    remain_Yangtze_River_Delta = []
    remain_other = []
    for row in remain.itertuples():
        province = getattr(row, 'province')
        if province in Yangtze_River_Delta:
            remain_Yangtze_River_Delta.append(row)
        else:
            remain_other.append(row)
    remain_Yangtze_River_Delta = pd.DataFrame(remain_Yangtze_River_Delta)
    remain_other = pd.DataFrame(remain_other)
    print('50之后 长三角地区：', len(remain_Yangtze_River_Delta))
    print('50之后 其余地区：', len(remain_other))

    # save results
    top_Yangtze_River_Delta.iloc[:, 2:4].to_csv(
        './dataResult/前50_长三角地区.csv', index=False)
    top_other.iloc[:, 2:4].to_csv('./dataResult/前50_其余地区.csv', index=False)
    remain_Yangtze_River_Delta.iloc[:, 2:4].to_csv(
        './dataResult/50之后_长三角地区.csv', index=False)
    remain_other.iloc[:, 2:4].to_csv('./dataResult/50之后_其余地区.csv', index=False)
