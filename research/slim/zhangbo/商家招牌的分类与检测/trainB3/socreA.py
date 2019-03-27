import pandas as pd
ip1='../baidu/datasets/result.csv'
fn='resA1'
# fn='res12'
ip='commit/%s.csv'%fn
df1=pd.read_csv(ip1,header=None)
df2=pd.read_csv(ip,header=None,sep=' ')
df1.columns=['id','y']
df2.columns=['id','p']
print(df1.shape,df2.shape)
df=pd.merge(df1,df2,on=['id'])
print(df.shape)
df['f']=df['y']-df['p']
print(df[df['f']==0].shape)