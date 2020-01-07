import numpy as np 
import pandas as pd 
from datetime import datetime


#---------定义层次分析法权重---------
#一级权重
weight=np.array([0.2,0.3,0.1,0.4])
#商务标权重系数,other合并到negotiationpercent中,合并后为0.2，暂时删除negotiationpercent,暂时删除公司资质
weightCommerce=np.array([0.1,0.2,0.3,0.1])/0.7
#技术标权重系数,去除服务半径
weightTech=np.array([0.1,0.15,0.25,0.3])/0.8
#投标价权重系数,去除议标价格
weightTenderprice=np.array([1])
#综合信誉权重系数,去除平台保证金
weightCredit=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.15,0.05,0.05])/0.85
#其他指标权重系数，去除
# weightOther=[]


#--------------计算函数----------------

def computeScore(data):
    commerce=np.array(data[[1,2,3,4]],dtype='float')
    tech=np.array(data[[5,6,7,8]],dtype='float')
    tenderPrice=np.array(data[[9]],dtype='float')
    credit=np.array(data[[10,11,12,13,14,15,16,17,18]],dtype='float')

    #commerce
    commerce[:,0]=commerce[:,0]*5
    commerce[np.where(commerce[:,0]>100),0]=100

    commerce[np.where(commerce[:,1]==1),1]=100
    commerce[np.where(commerce[:,1]==2),1]=50
    commerce[np.where(commerce[:,1]==3),1]=0

    commerce[np.where(commerce[:,2]<=1000),2]=10
    commerce[np.where((commerce[:,2]>1000)&(commerce[:,2]<=10000)),2]=30
    commerce[np.where((commerce[:,2]>10000)&(commerce[:,2]<=100000)),2]=60
    commerce[np.where((commerce[:,2]>100000)&(commerce[:,2]<=500000)),2]=80
    commerce[np.where(commerce[:,2]>500000),2]=100

    commerce[:,3]*=20

    #tech
    tech[:,0:3]*=20
    max=tech[:,3].max()
    min=tech[:,3].min()
    difference=max-min
    tech[:,3]=100*(tech[:,3]-min)/difference

    #tenderPrice
    tenderPrice[:,0]*=20

    #credit
    for i in range(6):
        max=credit[:,i].max()
        min=credit[:,i].min()
        difference=max-min
        credit[:,i]=100*(credit[:,i]-min)/difference

    credit[:,6]*=2
    credit[:,7]*=100
    credit[:,8]*=20

    scoreCommerce=np.dot(commerce,weightCommerce)
    scoreTech=np.dot(tech,weightTech)
    scoreTenderPrice=np.dot(tenderPrice,weightTenderprice)
    scoreCredit=np.dot(credit,weightCredit)

    score4=np.array([scoreCommerce,scoreTech,scoreTenderPrice,scoreCredit]).T
    score=np.dot(score4,weight)

    max=score.max()
    min=score.min()
    difference=max-min
    score=100*(score-min)/difference
    scoreForAnalysis=score
    score=pd.DataFrame(score)

    return score

def computeTime(dataTime):
    time=[]
    timeBegin=datetime(2015,4,8)
    endTime=datetime(2019,12,31)
    dataTime.columns=['regchecktime','createtime']
    for index,row in dataTime.iterrows():
        if isinstance(row['regchecktime'],str) and isinstance(row['createtime'],str):
            # time.append((endTime.year-timeBegin.year)*12+endTime.month-timeBegin.month)
            time.append((endTime-timeBegin).days)
        elif isinstance(row['regchecktime'],str) and isinstance(row['createtime'],datetime):
            # time.append((endTime.year-row['createtime'].year)*12+endTime.month-row['createtime'].month)
            time.append((endTime-row['createtime']).days)
        else:
            # time.append((endTime.year-row['regcherktime'].year)*12+endTime.month-row['regcherktime'].month)
            time.append((endTime-row['regchecktime']).days)

    time=np.array(time)
    time=time/365
    
    return pd.DataFrame(time)


def computeActiveness(analyseDF,w1=0.7,w2=0.3):
    data_activeness=analyseDF[['tender','te-se','concern','con-se','log','log-se','score','time']]

    data_activeness=np.array(data_activeness)
    # tenderZero=(data_activeness[:,0]==0)
    tenderNotZero=(data_activeness[:,0]!=0)
    # logZero=(data_activeness[:,4]==0)
    logNotZero=(data_activeness[:,4]!=0)
    activenessTender=np.zeros(data_activeness.shape[0])
    activenessLog=np.zeros(data_activeness.shape[0])
    activenessTender[tenderNotZero]=w1*data_activeness[tenderNotZero,1]/(data_activeness[tenderNotZero,0]/data_activeness[tenderNotZero,7]/2)
    activenessLog[logNotZero]=w2*data_activeness[logNotZero,5]/(data_activeness[logNotZero,4]/data_activeness[logNotZero,7]/2)
    activeness=activenessTender+activenessLog
    activeness=pd.DataFrame(activeness)
    # activeness=pd.concat([Id,activeness],axis=1)
    activeness.columns=['activeness']

    return activeness