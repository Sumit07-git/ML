import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import seaborn as sns

df=fetch_california_housing()
dataset=pd.DataFrame(df.data)
print(dataset.head())

dataset.columns=df.feature_names
print(dataset.head())

print(df.target.shape)
dataset['price']=df.target
print(dataset.head())

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#linear regression

lin_regressor=LinearRegression()
mese=cross_val_score(lin_regressor,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mese=np.mean(mese)
print(mean_mese)

#ridge regression
ridge=Ridge()
parameter={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameter,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

#lasso regression

lasso=Lasso()
parameter={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameter,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
prediction_lasso=lasso_regressor.predict(X_test)
prediction_ridge=ridge_regressor.predict(X_test)

sns.displot(y_test-prediction_lasso)
plt.show()

sns.displot(y_test-prediction_ridge)
plt.show()





