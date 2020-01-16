import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from model import LogisticRegression
from utils import readDataFromMysql
from sklearn import preprocessing

# load data
data = readDataFromMysql('select zjc_supplier.id,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2;')
data_param = data.iloc[:, 1:]
data_param = np.array(data_param, dtype='float32')

# standardlize
scaler = preprocessing.StandardScaler().fit(data_param)
data_standard = scaler.transform(data_param)

'''The only one you need to define'''
# labels
labels = torch.rand(data_standard.shape[0],1).ge(0.5).float()

cuda=True if torch.cuda.is_available() else False

logistic_model = LogisticRegression()

# 定义损失函数和优化器
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)
criterion = nn.BCELoss()

if cuda:
    logistic_model.cuda()
    criterion.cuda()

FloatTensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor
data_standard=Variable(FloatTensor(data_standard))

print("training...")
begin = datetime.now()

# 开始训练
for iteration in range(10000):

    out = logistic_model(data_standard)
    loss = criterion(out, labels)
    print_loss = loss.data.item()
    mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
    correct = (mask == labels).sum()  # 计算正确预测的样本个数
    acc = correct.item() / data_standard.size(0)  # 计算精度
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 每隔5000iteration打印一下当前的误差和精度
    if (iteration + 1) % 5000 == 0:
        print('*'*10)
        print('iteration {}'.format(iteration+1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        print('acc is {:.4f}'.format(acc))  # 精度

end = datetime.now()

print("total time:", (end-begin).seconds)
