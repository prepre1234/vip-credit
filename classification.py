import pandas as pd 
import numpy as np 
# from matplotlib import pyplot as plt

from utils import *
from compute import *


# data=readDataFromMysql('select * from zjc_supplier;')
data=readDataFromMysql('select zjc_supplier.id,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average,zjc_supplier.regchecktime,zjc_supplier.createtime from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2;')


'''
# result.query('score<10')
commerce=pd.DataFrame(commerce)
tech=pd.DataFrame(tech)
tenderPrice=pd.DataFrame(tenderPrice)
credit=pd.DataFrame(credit)
#供应商在各个标准上的得分
dataScore=pd.concat([id,commerce,tech,tenderPrice,credit],axis=1)
dataScore.columns=list(range(19))
'''

#层次分析法计算每个供应商分数
score=computeScore(data)
#计算每个供应商的注册/审核通过的时间
time=computeTime(data[[19,20]])

#analyseDF：整理后可供二次分析的数据
analyseDF=data[[0,10,11,12,13,14,15]]
analyseDF=pd.concat([analyseDF,score,time],axis=1)
analyseDF.columns=['id','tender','te-se','concern','con-se','log','log-se','score','time']

#计算每个供应商的活跃度
activeness=computeActiveness(analyseDF,0.7,0.3)

#analyseDF加上活跃度
analyseDF=pd.concat([analyseDF,activeness],axis=1)



###########
##数据分析##   整体分析
###########

#%%
#--------------------供应商总体分布情况[各个分数段分布]--------------------
distribute_list=[]
for i in range(10):
    query='score>='+str(i*10)+' and score<'+str((i+1)*10)
    distribute_list.append(len(analyseDF.query(query)))

analyseDF[['score']].to_csv('./distribution_all.csv')

#%%
#----------去除最高的百位供应商后剩余供应商归一化（0~100）后的整体分布[各个分数段分布]------------
residual_analyseDF=analyseDF.query('score<=40')
score_=np.array(residual_analyseDF[['score']]).squeeze()
max=score_.max()
min=score_.min()
difference=max-min
score_=100*(score_-min)/difference
score_=pd.DataFrame(score_)
score_.columns=['score']
score_.to_csv('./distribution_residual.csv')

residual_distribute_list=[]
for i in range(10):
    query='score>='+str(i*10)+' and score<'+str((i+1)*10)
    distribute_list.append(len(residual_analyseDF.query(query)))

#%%
#----------供应商总体分布[新/老低/老高 三个类别的数量]---------
new_oldHigh_oldLow=[len(analyseDF.query('time<=0.5')),len(analyseDF.query('time>0.5 and score<=20')),len(analyseDF.query('time>0.5 and score>20'))]

#%%
#-----------------高分老供应商活跃度统计-------------
old_high=analyseDF.query('time > 0.5 and score > 20')
sleep=old_high.query('activeness <= 0.1')
toSleep=old_high.query('activeness > 0.1 and activeness <= 0.4')
active=old_high.query('activeness > 0.4')
sleep_toSleep_active=[len(sleep),len(toSleep),len(active)]

#%%
###########
##数据分析##   按id分析
###########

#--------------------提供供应商id/name---------------------


#2019年投标/中标供应商分析
id_list=readDataFromMysql("SELECT DISTINCT(supplier_id) from zjc_bid_tender where tendertime>'2019'")
id_list=list(id_list[0])

'''
#已充值的vip供应商
id_list=pd.read_csv('supplier_vip.txt')
id_list=list(np.array(id_list).flatten())
'''

#--------------------分析---------------------
search=analyseDF.loc[analyseDF['id'].isin(id_list)]
#未查询到的供应商
no_results=list(set(id_list).difference(set(search['id'])))

#新供应商
suppliers_new=search.loc[search['time'] <= 0.5]
#老供应商中低分的
suppliers_old_lowScore=search.loc[(search['time']>0.5) & (search['score']<=20)]
#老供应商中高分的
suppliers_old_highScore=search.loc[(search['time']>0.5) & (search['score']>20)]

#新/老低/老高分布统计
new_oldLow_oldHigh=[len(suppliers_new),len(suppliers_old_lowScore),len(suppliers_old_highScore)]

#高分供应商的活跃度
sleep=suppliers_old_highScore.query('activeness<=0.1')
toSleep=suppliers_old_highScore.query('activeness>0.1 and activeness<=0.4')
active=suppliers_old_highScore.query('activeness>0.4')
#高分老供应商活跃度统计
sleep_toSleep_active=[len(sleep),len(toSleep),len(active)]


# %%
