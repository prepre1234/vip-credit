import pandas as pd 
import pymysql

def readDataFromMysql(sql_order):
    '''
    查询
    '''

    db=pymysql.connect('localhost','root','Qq56138669','zjc')
    cursor=db.cursor()
    try:
        cursor.execute(sql_order)
        data=cursor.fetchall()
        data=pd.DataFrame(list(data))
    except:
        data=pd.DataFrame()
    db.close()
    return data

