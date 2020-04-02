import pandas as pd
import numpy as np
import pymysql


def get_data(data_pd, label):
    db = pymysql.connect('localhost', 'root', 'Qq56138669', 'zjc')
    cursor = db.cursor()
    # data_list = []
    for row in data_pd.itertuples():
        name = getattr(row, 'name')
        sql_order = "select zjc_supplier.id,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2 and zjc_supplier.name='"+name+"';"

        try:
            cursor.execute(sql_order)
            data = list(cursor.fetchone())
            data.insert(0, label)
            yield tuple(data)
            # data_list.append(data)
        except:
            data = ()
    db.close()
    # return pd.DataFrame(data_list)


data_pay = pd.read_csv("./data/vip_1.csv")
data_refuse = pd.read_csv("./data/vip_0.csv")

data_pay = pd.DataFrame([data for data in get_data(data_pay, 1)])
data_refuse = pd.DataFrame([data for data in get_data(data_refuse, 0)])

pd.concat([data_pay, data_refuse]).to_csv("./dataDump/vip_data.csv")
