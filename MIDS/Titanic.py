import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

titanic = pd.read_csv('train.csv')
titanic.head()

titanic.tail()

titanic.describe()

titanic.columns

titanic.dtypes

column_names = titanic.columns
for column in column_names:
    print(column + ' - ' + str(titanic[column].isnull().sum()))

titanic.head()

titanic= titanic.drop(['PassengerId'], axis=1)
titanic = titanic.drop(['Name'], axis=1)

titanic = titanic.drop(['Cabin','Ticket'], axis=1)

titanic = titanic.drop(['Fare'], axis=1)

titanic

titanic.Survived.value_counts()