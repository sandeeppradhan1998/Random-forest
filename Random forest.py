# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 02:49:28 2019

@author: Dilip
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from sklearn import preprocessing
#le=preprocessing.LabelEncoder
#nec=preprocessing.OneHotEncoder
#y=y.apply(le.fit_transform)
#y=['variety',1]


#import dataset
dataframe=pd.read_csv("Position_Salaries.csv")
x=dataframe.iloc[:,1:2].values
y=dataframe.iloc[:,2].values

#split the dataset
from sklearn.model_selection import train_test_split

#import Random forest 
from sklearn.ensemble import RandomForestRegressor
r_regression= RandomForestRegressor(n_estimators=10,random_state=0)
r_regression.fit(x,y)
r_pred=r_regression.predict(x)


# Visualising the Random Forest Regression results (higher resolution)
x_meet = np.arange(min(x), max(x), 0.01)
x_meet = x_meet.reshape((len(x_meet), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_meet, r_regression.predict(x_meet), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()