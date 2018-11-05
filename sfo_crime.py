import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import datasets 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("../input/train.csv")
df2 = df['Category']

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

X = dummmy_df(df, todummy_list)
lb = preprocessing.LabelEncoder()
y = lb.fit_transform(df2)

df_test = pd.read_csv("../input/test.csv")

df_test['Dates'] = pd.to_datetime(df_test['Dates'])
df_test['Counter'] = 1
df_test['Simul_Crimes'] = df_test.groupby(['Address', 'Dates'])['Counter'].transform('count')
df_test['Year'], df_test['Month'],df_test['Hour'] = df_test['Dates'].dt.year, df_test['Dates'].dt.month,df_test['Dates'].dt.hour

df_test = df_test[['Year', 'Month', 'Hour', 'DayOfWeek', 'PdDistrict', 'Address', 'Simul_Crimes', 'X', 'Y', 'Id']]

df_test['Street'] = df_test.apply(street, axis=1) 
    
df_test['Season'] = df_test.apply(season, axis=1)

df_test['Evening'] = df_test.apply(evening, axis=1)

df_test = df_test.drop(['Address'], axis=1)

IQR = df_test.X.quantile(0.75) - df_test.X.quantile(0.25)
Lower_fence_X = df_test.X.quantile(0.25) - (IQR * 3)
Upper_fence_X = df_test.X.quantile(0.75) + (IQR * 3)
df_test.loc[df_test.X < -122.51093037786198, 'X']= -122.51093037786198
df_test.loc[df_test.X > -122.32897987265702, 'X']= -122.32897987265702

IQR = df_test.Y.quantile(0.75) - df_test.Y.quantile(0.25)
Lower_fence_Y = df_test.Y.quantile(0.25) - (IQR * 3)
Upper_fence_Y = df_test.Y.quantile(0.75) + (IQR * 3)
df_test.loc[df_test.Y > 37.8801919977151, 'Y']= 37.8801919977151

todummy_list = ['Year', 'Month', 'Hour', 'DayOfWeek', 'PdDistrict', 'Street', 'Season', 'Evening', 'Simul_Crimes']

X_df_test = dummmy_df(df_test, todummy_list)

missing_cols = set( X.columns ) - set( X_df_test.columns )

for c in missing_cols:
    X_df_test[c] = 0

X_df_test = X_df_test[X.columns]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaler.fit(X_df_test) 

# Validation Testing Code
#------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.90, test_size=0.10)


# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
# params = {'max_depth':5, 'silent':1, 'eta':0.4, 'objective':'multi:softprob', 'sub_sample':0.9, 'num_class':39, 'eval_metric':'mlogloss'}
# watchlist = [(dtrain, 'train'), (dtest, 'val')]
# bst = xgb.train(params, dtrain, 50, watchlist)

params = {'max_depth':5, 'silent':1, 'eta':0.4, 'objective':'multi:softprob', 'sub_sample':0.9, 'num_class':36, 'eval_metric':'mlogloss'}

xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(X_df_test)
bst = xgb.train(params, xgb_train, 50)

predicts = bst.predict(xgb_test)

submission_results = pd.DataFrame(predicts, columns=lb.classes_)
submission_results['Id'] = df_test['Id']

cols = submission_results.columns.tolist()
cols.insert(0, cols.pop(cols.index('Id')))
submission_results = submission_results.reindex(columns= cols)

submission_results.to_csv('submission_xgb.csv', index=False)
