import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df=pd.read_csv('iris.csv')
print(df.head())
print(df.shape)

#univariate analysis

df_setosa=df.loc[df['variety']=='Setosa']
print(df_setosa)
df_versicolor=df.loc[df['variety']=='Versicolor']
print(df_versicolor)
df_virginica=df.loc[df['variety']=='Virginica']
print(df_virginica)

plt.plot(df_setosa['sepal.length'],np.zeros_like(df_setosa['sepal.length']),'o')
plt.plot(df_versicolor['sepal.length'],np.zeros_like(df_versicolor['sepal.length']),'o')
plt.plot(df_virginica['sepal.length'],np.zeros_like(df_virginica['sepal.length']),'o')
plt.xlabel('Sepal Length')
plt.show()


#bivariate analysis

sns.FacetGrid(df,hue="variety",height=5).map(plt.scatter,"sepal.length","sepal.width").add_legend();
plt.show()

#multivariate analysis

sns.pairplot(df,hue="variety",height=5)
plt.show()

#histogram

sns.histplot(data=df,x="sepal.length",hue="variety",bins=20,kde=True,palette="Set2")
plt.title("Sepal Length distribution by species")
plt.show()

#Z-score

z_scores=stats.zscore(df.select_dtypes(include=['float64','int64']))
z_scores_df=pd.DataFrame(z_scores,columns=df.select_dtypes(include=['float64','int64']).columns)
print(z_scores_df.head())