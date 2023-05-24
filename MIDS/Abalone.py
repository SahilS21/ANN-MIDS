import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df= pd.read_csv('abalone.csv')

df["Sex"]= pd.factorize(df['Sex'])[0]
df.head()

x_train, x_test, y_train, y_test= train_test_split(df.drop("Rings",axis= 1),df["Rings"], test_size=0.3, random_state= 4 )

k= 5
knn= KNeighborsClassifier(n_neighbors= k)
knn.fit(x_train, y_train)

y_pred= knn.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score

accuracy= accuracy_score(y_test, y_pred)
print(accuracy)

a_train, a_test, b_train, b_test= train_test_split(df.drop("Rings", axis= 1),df["Rings"]+ 1.5, test_size= 0.2, random_state= 5)

model= LinearRegression()
model.fit(a_train,b_train)

b_pred= model.predict(a_test)
print(b_pred)