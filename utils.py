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


'''
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
'''


def computeTime(rc_time):
    time_init = datetime(2015, 4, 8)
    endTime = datetime.today()

    rc_time.columns = ['regchecktime', 'createtime']
    rc_time.loc[rc_time['regchecktime'] ==
                '0000-00-00 00:00:00', 'regchecktime'] = time_init
    rc_time.loc[rc_time['createtime'] ==
                '0000-00-00 00:00:00', 'createtime'] = time_init

    time = pd.DataFrame(rc_time.max(axis=1))
    days = (endTime-time).apply(lambda x: x[0].days, axis=1)
    years = np.array(days)/365

    return pd.DataFrame(years)
