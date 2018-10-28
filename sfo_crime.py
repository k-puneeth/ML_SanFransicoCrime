# Next Up
# 1. [x] Random Forest with Hyperparameter Tuning 
# 2. [x] Try using PCA and Feature Selection (Makes it worse)
# 3. [] Add Features Street Suffix, Season, Coordinate, Simultaneous Crimes
# 4. [] Explore Interactions (especially between Season-Month and Coordinate-PdDistrict)
# 5. [] Make a submission
# 6. [] Train other models (Logistic Regression, Naive Bayes, SVM)
import numpy as np
import pandas as pd
from sklearn import datasets 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

df = pd.read_csv("train.csv")
df = df.sample(frac=1)
df2 = df['Category']

df = df.drop(['Id', 'Resolution','Descript'],axis=1)
df['Dates'] = pd.to_datetime(df['Dates'])
df['Year'], df['Month'],df['Hour'] = df['Dates'].dt.year, df['Dates'].dt.month,df['Dates'].dt.hour
df=df.drop(['Dates','X','Y'],axis=1)
df = df[['Year', 'Month', 'Hour', 'DayOfWeek', 'PdDistrict', 'Address']]
df = df.drop(['Address'], axis=1)

# Features to add : Street Suffix, Simultaneous Crimes, Coordinate, Season

todummy_list = ['Year', 'Month', 'Hour', 'DayOfWeek', 'PdDistrict']

def dummmy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis = 1)
    return df

X = dummmy_df(df, todummy_list)[:100000]  # Using only 100k of the data as 800k overflows
lb = preprocessing.LabelEncoder()
y = lb.fit_transform(df2)[:100000]


# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# X = pd.DataFrame(pca.fit_transform(X))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.60, test_size=.40, random_state=10)

# import sklearn.feature_selection

# select = sklearn.feature_selection.SelectKBest(k='all')
# selected_features = select.fit(X_train, y_train)
# indices_selected = selected_features.get_support(indices=True)
# colnames_selected = [X.columns[i] for i in indices_selected]

# X_train = X_train[colnames_selected]
# X_test = X_test[colnames_selected]

rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=10, min_samples_leaf=2, random_state=20)

rf_model.fit(X_train, y_train)

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
