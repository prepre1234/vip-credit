import pandas as pd
import pymysql
import numpy as np
from datetime import datetime


def readDataFromMysql(sql_order):
    '''
    查询
    '''

    db = pymysql.connect('localhost', 'root', 'Qq56138669', 'zjc')
    cursor = db.cursor()
    try:
        cursor.execute(sql_order)
        data = cursor.fetchall()
        data = pd.DataFrame(list(data))
    except:
        data = pd.DataFrame()
    db.close()
    return data


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
