import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df= pd.read_csv('placement.csv')
print(df.head())


plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package (in lpa)')
plt.show()

X=df.iloc[:,0:1]
y=df.iloc[:,-1]

print(X)
print(y)


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=2)
Ir= LinearRegression()

Ir.fit(X_train,y_train)

LinearRegression()

print(X_test)
print(y_test)


pred=Ir.predict(X_test.iloc[[1]])
print(pred)

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,Ir.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package (in lpa)')
plt.show()

print(Ir.coef_) #value of m

print(Ir.intercept_) #b




