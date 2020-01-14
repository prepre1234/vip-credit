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


def readDataFromMysql_one_apiece(sql_order,search_list,mode=1):
    '''
    从列表里一个一个查询数据
    mode=1，按id查
    mode=2，按name查
    '''

    db=pymysql.connect('localhost','root','Qq56138669','zjc')
    cursor=db.cursor()
    data=[]
    try:
        if mode==1:
            for x in search_list:
                sql=sql_order+"'"+x+"'"+';'
                cursor.execute(sql)
                data_one=cursor.fetchall()
                data.append(data_one[0])
        elif mode==2:
            for x in search_list:
                sql=sql_order+x+';'
                cursor.execute(sql)
                data_one=cursor.fetchall()
                data.append(data_one[0])
        else:
            raise Exception("mode error,must be 1 or 2!")
    except:
        data=tuple()
    db.close()
    
    return pd.DataFrame(list(data))

