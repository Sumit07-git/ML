import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

X=6 * np.random.rand(200,1)-3
y=0.8 * X**2+0.9  * X+2+np.random.randn(200,1)

plt.plot(X,y,'b.')
plt.xlabel("X")
plt.ylabel("y")
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

lr=LinearRegression()
lr.fit(X_train,y_train)
# LinearRegression()

y_pred=lr.predict(X_test)
fd=r2_score(y_test,y_pred)
print(fd)

plt.plot(X_train,lr.predict(X_train),color='r')
plt.plot(X,y,"b.")
plt.xlabel("X")
plt.ylabel("y")
plt.show()