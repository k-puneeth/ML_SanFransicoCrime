import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
# print(df.isnull().any())
df = df.drop(['Id','Resolution','Descript'],axis=1)
# print(df.head())
df['Dates'] = pd.to_datetime(df['Dates'])
df['year'], df['month'],df['hour'] = df['Dates'].dt.year, df['Dates'].dt.month,df['Dates'].dt.hour
df=df.drop(['Dates','X','Y'],axis=1)
# print(df['Address'].unique())
# print(df['Category'].unique())
dummies_dow = pd.get_dummies(df.DayOfWeek)
df = pd.concat([df,dummies_dow],axis='columns')
df = df.drop(['DayOfWeek'],axis=1)
print(df.head())
