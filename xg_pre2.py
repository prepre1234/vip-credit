from utils import readDataFromMysql
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import xgboost as xgb
from utils import*
from matplotlib import pyplot
from datetime import date
from datetime import datetime
import time


#公司id、登录次数、投标次数、关注次数、创建时间
sql = 'select supplier_id,login_count,tender_count,collect_count,time from zjc_supplier_day_log'
all_supplier = readDataFromMysql(sql)
all_supplier.columns =['id','l_c','t_c','c_c','time']
#print(all_supplier)

def parser(x):
    return datetime.strptime('%Y-%m-%d')

#平方系数加权移动平均线
def NMT(df_1):    #df_1传入格式'id','data','time'
    col_liu=[] 
    col_id=[]
    for i in df_1['id'].drop_duplicates():
        data=df_1.loc[df_1['id']==i]
        df1=data.iloc[:,-2].values.tolist() 
        sum_fz=0
        sum_fm=0
        for j in range(data.shape[0]):
            sum_fz=sum_fz+df1[j]*(j+1)*(j+1) #
            sum_fm=sum_fm+(j+1)*(j+1)

        col_liu.append(sum_fz/sum_fm)
        col_id.append(i)

    col_liu=pd.DataFrame(col_liu)
    col_liu.columns=['pred']
    col_id=pd.DataFrame(col_id)
    col_id.columns=['id']

    result=pd.concat([col_id,col_liu],join='inner',axis=1)
    print(result)

#把datetime类型转外时间戳形式
def datetime_toTimestamp(dateTim):
    return time.mktime(dateTim.timetuple())

#'环比'增长率概念
def CRR(df_1):
    col_liu=[] 
    col_id=[]
    for i in df_1['id'].drop_duplicates():
        data=df_1.loc[df_1['id']==i]
        today = int(time.time()/(3600*24)) #转化成时间戳
        midday=today-180 
        foreday=today-365
        df1=data.iloc[:,-2].values.tolist() #data换成列表形式
        df2=data.iloc[:,-1].values.tolist() #time
        s1=s2=0
        df22=[] #临时存储time的数值型
        for j in range(data.shape[0]):
            temp=int(datetime_toTimestamp(df2[j])/(3600*24))
            df22.append(temp)
            if df22[j]<=today&df22[j]>midday: #最近半年累积
                s1=s1+df1[j]
            elif df22[j]<=midday&df22[j]>foreday: #之前半年累积
                s2=s2+df1[j]
        if s2!=0:    
            hb=(s1-s2)/s2
        else:
            hb=0
        col_id.append(i)
        col_liu.append(hb)

    col_liu=pd.DataFrame(col_liu)
    col_liu.columns=['hb']
    col_id=pd.DataFrame(col_id)
    col_id.columns=['id']

    result=pd.concat([col_id,col_liu],join='inner',axis=1)
    return result



df=all_supplier[['id','t_c','time']] #中间换成l_c和t_c和c_c
#df=df.iloc[0:100] #可以注销掉，取100个运行比较快
re=CRR(df)
re.to_csv('./data/jry.csv', index=False)
