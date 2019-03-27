from sklearn.naive_bayes import MultinomialNB
import pandas as pd
ip='output/'
order=1
op='commit/'
train=pd.read_csv(ip+'val%d.csv'%order)
test=pd.read_csv(ip+'test%d.csv'%order)
L=['y1','y2','y3']
label=['label']
clf = MultinomialNB()
clf.fit(train[L].values, train[label].values)

res=pd.DataFrame()
res['id']=test['id']
res['label']=clf.predict(test[L].values)
res.to_csv(op+'resA%d.csv'%order,sep=' ',header=None,index=False)
