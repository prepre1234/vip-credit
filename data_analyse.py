import pandas as pd

pd_250 = pd.read_csv('dataResult/total.csv')
pd_newdata = pd.read_csv('data/new_data.csv')
pd_newadd = pd_newdata[109:]
pd_newadd.to_csv('data/new_add.csv', index=False)

# 所有的前250个预测名单的set
pd_250_set = set(pd_250.iloc[:, 0])

# 本次最新数据统计新增的VIP供应商在这250家里面的有多少
num = 0
for row in pd_newadd.itertuples():
    name = getattr(row, 'name')
    if name in pd_250_set:
        num = num+1
print(num)
