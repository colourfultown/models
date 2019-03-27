import pandas as pd

op='commit/'
ip1 = '../createDataB/output/testA1/'
name=pd.read_csv(ip1+'filename.csv')
fn='resB2.txt'
ip2='result/%s'%fn
res=pd.read_csv(ip2)
res=pd.merge(name,res,on=['code'])
res[['name','pre']].to_csv(op+fn.replace('.txt','.csv'),index=False,header=None,sep=' ')