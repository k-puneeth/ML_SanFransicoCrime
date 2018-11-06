# Next Up
# 1. [x] Make a submission
# 2. [] Train other models (Logistic Regression, Naive Bayes, SVM)

import numpy as np
import pandas as pd
import xgboost as xgb
import s2sphere as sp
from sklearn import datasets 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split, cross_val_score

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

train_len = len(df_train)

df = pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)

df2 = df_train['Category']

df = df.drop(['Id', 'Resolution','Descript'],axis=1)

df['Dates'] = pd.to_datetime(df['Dates'])
df['Counter'] = 1
df['Simul_Crimes'] = df.groupby(['Address', 'Dates'])['Counter'].transform('count')
df['Year'], df['Month'],df['Hour'] = df['Dates'].dt.year, df['Dates'].dt.month,df['Dates'].dt.hour

df = df[['Year', 'Month', 'Hour', 'DayOfWeek', 'PdDistrict', 'Address', 'Simul_Crimes', 'X', 'Y']]


def street(x):
    if "/" in x['Address']:
        return 'II'
    elif "HWY" in x['Address']:
        return "HY"
    elif "WAY" in x['Address']:
        return "WY"
    else :
        return x['Address'][-2] + x['Address'][-1]

df['Street'] = df.apply(street, axis=1) 

def season(x):
    if (x['Month'] >=6 and x['Month']<=8):
        return 'Summer'
    elif (x['Month'] >=9 and x['Month']<=11):
        return 'Autumn'
    elif (x['Month']>=3 and x['Month']<=5):
        return 'Spring'
    else :
        return 'Winter'
    
df['Season'] = df.apply(season, axis=1)

def evening(x):
    if x['Hour'] in range(18, 23):
        return 1
    else:
        return 0

df['Evening'] = df.apply(evening, axis=1)

df = df.drop(['Address'], axis=1)

IQR = df.X.quantile(0.75) - df.X.quantile(0.25)
Lower_fence_X = df.X.quantile(0.25) - (IQR * 3)
Upper_fence_X = df.X.quantile(0.75) + (IQR * 3)
df.loc[df.X < -122.51093037786198, 'X']= -122.51093037786198
df.loc[df.X > -122.32897987265702, 'X']= -122.32897987265702

IQR = df.Y.quantile(0.75) - df.Y.quantile(0.25)
Lower_fence_Y = df.Y.quantile(0.25) - (IQR * 3)
Upper_fence_Y = df.Y.quantile(0.75) + (IQR * 3)
df.loc[df.Y > 37.8801919977151, 'Y']= 37.8801919977151

todummy_list = ['Year', 'Month', 'Hour', 'DayOfWeek', 'PdDistrict', 'Street', 'Simul_Crimes', 'Season', 'Evening']

def dummmy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis = 1)
    return df

df['Coordinate'] = df[['X', 'Y']].apply(lambda x: sp.CellId.from_lat_lng(sp.CellId.from_degrees(x[0], x[1])).level(), axis=1)
df = df.drop(['X', 'Y'], axis=1)

X = dummmy_df(df, todummy_list)
lb = preprocessing.LabelEncoder()
y = lb.fit_transform(df2)

X_train = X[:train_len]
X_test = X[train_len:]



params = {'max_depth':5, 'silent':1, 'eta':0.4, 'objective':'multi:softprob', 'sub_sample':0.9, 'num_class':36, 'eval_metric':'mlogloss'}

xgb_train = xgb.DMatrix(X_train, label=y)
xgb_test = xgb.DMatrix(X_test)
bst = xgb.train(params, xgb_train, 70)

predicts = bst.predict(xgb_test)

submission_results = pd.DataFrame(predicts, columns=lb.classes_)
submission_results.to_csv('submission_xgb.csv', index_label = 'Id')
