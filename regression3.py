import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X,y= load_diabetes(return_X_y=True)

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=2)
# print(X_train.shape)
# print(y_train.shape)

reg=LinearRegression()

reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)

r2_score(y_test,y_pred)
# print(pd)

reg.coef_
# print(fd)

reg.intercept_
# print(pd)

class MeraLR:

    def __init__(self):
        self.coef_=None
        self.intercept_=None

    def fit(self,X_train,y_train):
        X_train=np.insert(X_train,0,1,axis=1)

        betas=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.intercept_=betas[0]
        self.coef_=betas[1:]
       

    def predict(self,X_test):
        y_pred=np.dot(X_test,self.coef_)+self.intercept_
        return y_pred



lr=MeraLR()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
R2=r2_score(y_test,y_pred)
print(R2)
     




