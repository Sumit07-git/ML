import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df= pd.read_csv('Advertising.csv', index_col=0)
X= df[['TV','Radio','Newspaper']]
y=df['Sales']
print(df.head())

X=sm.add_constant(X)
model=sm.OLS(y,X).fit()
print(model.summary())

print(X.iloc[:,1:].corr())

df_sal=pd.read_csv('Salary_Data.csv')
print(df_sal.head())

X1=df_sal[['YearsExperience','Age']]
y1=df_sal['Salary']

X1=sm.add_constant(X1)
model1=sm.OLS(y1,X1).fit()
print(model1.summary())

print(X1.iloc[:,1:].corr())








