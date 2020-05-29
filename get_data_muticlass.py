import pandas as pd
import numpy as np
import pymysql
from datetime import datetime
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity


def computeTime(dataTime):
    time = []
    timeBegin = datetime(2015, 4, 8)
    # endTime = datetime(2020, 12, 31)
    endTime = datetime.today()
    dataTime.columns = ['regchecktime', 'createtime']
    for index, row in dataTime.iterrows():
        if isinstance(row['regchecktime'], str) and isinstance(row['createtime'], str):
            time.append((endTime-timeBegin).days)
        elif isinstance(row['regchecktime'], str) and isinstance(row['createtime'], datetime):
            time.append((endTime-row['createtime']).days)
        else:
            time.append((endTime-row['regchecktime']).days)

    time = np.array(time)
    time = time/365

    return pd.DataFrame(time)


unfound = []


def getdata(data_pd, label):
    db = pymysql.connect('localhost', 'root', 'Qq56138669', 'zjc')
    cursor = db.cursor()
    # data_list = []
    for row in data_pd.itertuples():
        name = getattr(row, 'name')
        sql_order = "select zjc_supplier.name,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average,zjc_supplier.regchecktime,zjc_supplier.createtime from zjc_supplier,zjc_supplier_param where zjc_supplier.id=zjc_supplier_param.supplier_id and zjc_supplier.state=2 and zjc_supplier.name='"+name+"';"

        try:
            cursor.execute(sql_order)
            data = list(cursor.fetchone())
            data.insert(1, label)
            yield tuple(data)
            # data_list.append(data)
        except:
            data = ()
            if label == 1:
                print("正样本，未查到："+name)
            if label == 0:
                print("负样本，未查到："+name)
            unfound.append((name, label))
    db.close()
    # return pd.DataFrame(data_list)


data_pay = pd.read_csv("./data/vip_1.csv")
data_pay = data_pay.drop_duplicates()
data_refuse = pd.read_csv("./data/vip_0.csv")
data_refuse = data_refuse.drop_duplicates()

data_pay = pd.DataFrame([data for data in getdata(data_pay, 1)])
data_refuse = pd.DataFrame([data for data in getdata(data_refuse, 0)])

time = computeTime(data_pay.iloc[:, 20:])
data_pay = pd.concat([data_pay.iloc[:, 0:20], time], axis=1)
time = computeTime(data_refuse.iloc[:, 20:])
data_refuse = pd.concat([data_refuse.iloc[:, 0:20], time], axis=1)


data = pd.concat([data_pay, data_refuse])

data.columns = ['name', 'label', 'attach_count', 'type', 'fund', 'service_average', 'instock_service', 'instock_product', 'instock_deliverspeed', 'win', 'price',
                'tender', 'se_tender', 'bidconcern', 'se_bidconcern', 'login', 'se_login', 'integrity', 'contract_rate', 'instock_honesty', 'years']

new_supplier = data.query('years<0.5 and label ==1')
new_supplier.to_csv("./dataDump/new_supplier.csv", index=False)

unfound = pd.DataFrame(unfound, columns=['name', 'label'])
unfound.to_csv('./dataDump/unfound.csv', index=False)

old_supplier = data.query('years>=0.5')

# old_supplier_p=old_supplier.query('label==1')
# old_supplier_n=old_supplier.query('label==0')

name_label = old_supplier.iloc[:, :2]
remain = old_supplier.iloc[:, 2:]
num_of_p = len(name_label.query('label==1'))
num_of_n = len(name_label)-num_of_p

remain_scale = np.array(remain.astype('float32'))
# 标准化处理
scaler = preprocessing.StandardScaler().fit(remain_scale)
remain_scale = scaler.transform(remain_scale)
remain_scale_p = remain_scale[:num_of_p]
remain_scale_n = remain_scale[num_of_p:]

similarity_matrix = cosine_similarity(remain_scale_p, remain_scale_n)

index = np.where(similarity_matrix > 0.9)
index_p = np.unique(index[0])
index_n = np.unique(index[1])
index_all = np.concatenate((index_p, index_n+num_of_p))

old_supplier.iloc[index_all, 1] = 2
old_supplier.to_csv('./dataDump/vip_data.csv', index=False)
