import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import sklearn.model_selection import train_test_split

df= pd.read_csv('placement.csv')
print(df.head())


plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package (in lpa)')
plt.show()

x=df.iloc[:,0:1]
y=df.iloc[:,-1]

print(x)
print(y)


X_train,X_test,Y_train,Y_test= train_test_split()