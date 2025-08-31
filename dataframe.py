import pandas as pd
import numpy as np

df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=['Row1','Row2','Row3','Row4','Row5'],columns=["Column1","Column2","Column3","Column4"])
print(df.head())

# df.to_csv('Test1.csv')

print(df.loc['Row1'])
print(type(df.loc['Row1']))


print(df.iloc[:,1:].values)
print(df.iloc[:,1:].values.shape)

print(df.isnull().sum())  #checking null value in the dataset

print(df['Column1'].value_counts())
print(df['Column1'].unique())




