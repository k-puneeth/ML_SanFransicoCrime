# Next Up
# 1. [x] Make a submission
# 2. [] Train other models (Logistic Regression, Naive Bayes, SVM)
import numpy as np
import pandas as pd
from sklearn import datasets 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 


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
    
#df['Season'] = df.apply(season, axis=1)

def evening(x):
    if x['Hour'] in range(18, 23):
        return 1
    else:
        return 0

#df['Evening'] = df.apply(evening, axis=1)

df = df.drop(['Address'], axis=1)
#print(df)

IQR = df.X.quantile(0.75) - df.X.quantile(0.25)
Lower_fence_X = df.X.quantile(0.25) - (IQR * 3)
Upper_fence_X = df.X.quantile(0.75) + (IQR * 3)
df.loc[df.X < -122.51093037786198, 'X']= -122.51093037786198
df.loc[df.X > -122.32897987265702, 'X']= -122.32897987265702

# For Y
IQR = df.Y.quantile(0.75) - df.Y.quantile(0.25)
Lower_fence_Y = df.Y.quantile(0.25) - (IQR * 3)
Upper_fence_Y = df.Y.quantile(0.75) + (IQR * 3)
df.loc[df.Y > 37.8801919977151, 'Y']= 37.8801919977151

from sklearn.cluster import KMeans
#Cluster the data
kmeans = KMeans(n_clusters=40, random_state=0).fit(df[['X', 'Y']])
labels = kmeans.labels_

#Glue back to originaal data
df['Coordinate'] = labels

df.drop(['X', 'Y'], axis=1)

todummy_list = ['Year', 'Month', 'Hour', 'DayOfWeek', 'PdDistrict', 'Street', 'Simul_Crimes', 'Coordinate']

def dummmy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis = 1)
    return df

X = dummmy_df(df, todummy_list)
lb = preprocessing.LabelEncoder()
y = lb.fit_transform(df2)

#print(X.head())

# from sklearn.decomposition import PCA
# pca = PCA(n_components=20)
# X = pd.DataFrame(pca.fit_transform(X))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# import sklearn.feature_selection

# select = sklearn.feature_selection.SelectKBest(k='all')
# selected_features = select.fit(X_train, y_train)
# indices_selected = selected_features.get_support(indices=True)
# colnames_selected = [X.columns[i] for i in indices_selected]

# X_train = X_train[colnames_selected]
# X_test = X_test[colnames_selected]

rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=10, min_samples_leaf=1, random_state=50)

rf_model.fit(X, y)

predicted = rf_model.predict_proba(X_test)
predicted_loss = np.array(predicted)
print(log_loss(y_test, predicted_loss))

# For tuning hyperparameters
#---------------------------
# from sklearn.model_selection import GridSearchCV

# rfc_model = RandomForestClassifier(n_jobs=10, random_state=20, max_depth=20) 
# param_grid = {"n_estimators": [200, 250, 300], "min_samples_leaf" : [1, 2, 4]}
 
# rfc_grid = GridSearchCV(estimator=rfc_model, param_grid=param_grid, cv= 10)
# rfc_grid.fit(X_train, y_train)
# print(rfc_grid.best_params_)

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
#print(df_test)

IQR = df_test.X.quantile(0.75) - df_test.X.quantile(0.25)
Lower_fence_X = df_test.X.quantile(0.25) - (IQR * 3)
Upper_fence_X = df_test.X.quantile(0.75) + (IQR * 3)
df_test.loc[df_test.X < -122.51093037786198, 'X']= -122.51093037786198
df_test.loc[df_test.X > -122.32897987265702, 'X']= -122.32897987265702

# For Y
IQR = df_test.Y.quantile(0.75) - df_test.Y.quantile(0.25)
Lower_fence_Y = df_test.Y.quantile(0.25) - (IQR * 3)
Upper_fence_Y = df_test.Y.quantile(0.75) + (IQR * 3)
df_test.loc[df_test.Y > 37.8801919977151, 'Y']= 37.8801919977151

from sklearn.cluster import KMeans
#Cluster the data
kmeans = KMeans(n_clusters=40, random_state=0).fit(df_test[['X', 'Y']])
labels = kmeans.labels_

#Glue back to originaal data
df_test['Coordinate'] = labels

df_test.drop(['X', 'Y'], axis=1)

todummy_list = ['Year', 'Month', 'Hour', 'DayOfWeek', 'PdDistrict', 'Street', 'Season', 'Evening', 'Simul_Crimes', 'Coordinate']

X_df_test = dummmy_df(df_test, todummy_list)

missing_cols = set( X.columns ) - set( X_df_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_df_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_df_test = X_df_test[X.columns]

#print(X_df_test)
predicted_sub = rf_model.predict_proba(X_df_test)

submission_results = pd.DataFrame(predicted_sub, columns=lb.classes_)
submission_results['Id'] = df_test['Id']
cols = submission_results.columns.tolist()
cols.insert(0, cols.pop(cols.index('Id')))
submission_results = submission_results.reindex(columns= cols)
submission_results.to_csv('submission.csv', index=False)