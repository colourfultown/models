import pandas as pd
order=1

op='output/'
ip='input/'

df=pd.DataFrame()
for i in range(1,5):
    dt=pd.read_csv(ip+'test/resA%d.csv'%i,sep=' ',names=['id','y%d'%i])
    if len(df)==0:
        df=dt
    else:
        df=pd.merge(df,dt,on=['id'])
df.to_csv(op+'test%d.csv'%order,index=False)

df=pd.DataFrame()
for i in range(1,5):
    dt=pd.read_csv(ip+'val/B%d.txt'%i)
    dt.columns=['id','y%d'%i]
    if len(df)==0:
        df=dt
    else:
        df=pd.merge(df,dt,on=['id'])
dt=pd.read_csv('../createDataB/output/validaA3/valres.csv')
dt['label']=dt['label']+1
df=pd.merge(df,dt,on=['id'])
df.to_csv(op+'val%d.csv'%order,index=False)

