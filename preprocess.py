from sklearn import preprocessing
import pandas as pd
import numpy as np
import utils

import pickle

# load data
# data = readDataFromMysql('select zjc_supplier.id,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2;')
# data_param = data.iloc[:, 1:]
# data_param = np.array(data_param, dtype='float32')

# read data
data = pd.read_csv('./data/cs-training.csv').iloc[:, 1:]

# split label and data
label = data.iloc[:, 0]
data = np.array(data.iloc[:, 1:])

# standardlize
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)

# pickle&dump scaler
with open('./dataDump/scaler.pk', 'wb') as f:
    pickle.dump(scaler, f)

# recombine label and data
data = pd.concat([label, pd.DataFrame(data)], axis=1)
data[['SeriousDlqin2yrs']] = data[['SeriousDlqin2yrs']].astype('int64')

# dump scaler_data
data.to_csv('./dataDump/scaler_data.csv')
