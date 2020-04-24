import pandas as pd
import numpy as np
import pymysql
from datetime import datetime


def computeTime(dataTime):
    time = []
    timeBegin = datetime(2015, 4, 8)
    # endTime = datetime(2019, 12, 31)
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

pd.concat([data_pay, data_refuse]).to_csv("./dataDump/vip_data.csv")
