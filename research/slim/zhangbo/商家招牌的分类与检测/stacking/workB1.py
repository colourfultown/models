import xgboost as xgb
import pandas as pd


ip='output/'
order=1
op='commit/'
train=pd.read_csv(ip+'val%d.csv'%order)
test=pd.read_csv(ip+'test%d.csv'%order)
L=['y1','y2','y3','y4']
label=['label']
trX = train[L].values
trY = train[label].values
tsX = test[L].values

clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.01,
                                           n_estimators=5000, silent=True, objective="multi:softmax",
                                           colsample_bytree=0.8,subsample=0.8,
                                           nthread=-1, gamma=0.1, reg_alpha=0.05, reg_lambda=0.05, scale_pos_weight=1)
clf.fit(trX, trY, eval_metric="mlogloss", verbose=True, eval_set=[(trX, trY)],
                           early_stopping_rounds=10)

res=pd.DataFrame()
res['id']=test['id']
res['label']=clf.predict(tsX)
res.to_csv(op+'resB%d.csv'%order,sep=' ',header=None,index=False)
