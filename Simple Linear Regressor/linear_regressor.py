# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
'''
Importing data set
'''
data_set =  pd.read_csv("Salary_Data.csv")
'''
Getting Values from the CSV
1.Independent
2.Depentent
'''
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1:].values


'''
Train Test spliting.
'''
from sklearn.cross_validation import  train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

'''
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.fit_transform(X_test)
y_train = std_scaler.fit_transform(y_train)
y_test = std_scaler.fit_transform(y_test)
'''
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

'''
Ploting values.
'''

plt.scatter(X_train,y_train, color = 'green')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')

plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
