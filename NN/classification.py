import argparse
import sys

import numpy as np
from datetime import datetime
from sklearn import preprocessing

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import network
from utils import readDataFromMysql
from datasets import vipDataset

if sys.platform.find("win")!=-1:
    print(sys.platform)
    from colorama import init
    init(autoreset=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8,
                    help="size of the batches")
parser.add_argument("--epoch", type=int, default=500, help="size of epoches")
opt = parser.parse_args()
print(opt)

# load data
data = readDataFromMysql('select zjc_supplier.id,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2;')
data_param = data.iloc[:, 1:]
data_param = np.array(data_param, dtype='float32')

# standardlize
scaler = preprocessing.StandardScaler().fit(data_param)
data_standard = scaler.transform(data_param)

'''The only one you need to define'''
# labels
labels = torch.rand(data_standard.shape[0], 1).ge(0.5).float()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# network_model = network(data_standard.shape[1])
network_model = network()

# 定义损失函数和优化器
optimizer = torch.optim.Adam(
    network_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

if cuda:
    logistic_model.cuda()
    criterion.cuda()

print("training...")
begin = datetime.now()

# Training data loader
dataloader = DataLoader(
    vipDataset(data_standard, labels),
    batch_size=opt.batch_size,
    shuffle=True
)

# 开始训练
for epoch in range(opt.epoch):
    for i, batch in enumerate(dataloader):

        #data & labels
        data_train = Variable(Tensor(batch["data"]))
        labels_train = Variable(Tensor(batch["labels"]))

        optimizer.zero_grad()

        possibility = network_model(data_train)
        loss = criterion(possibility, labels_train)

        loss.backward()
        optimizer.step()
    sys.stdout.write("\r[Epoch \033[1;35m%d\033[0m/\033[1;31m%d\033[0m][loss: \033[1;33m%f\033[0m]" %
                     (epoch, opt.epoch, loss.item()))

end = datetime.now()

print("total time:", (end-begin).seconds)
