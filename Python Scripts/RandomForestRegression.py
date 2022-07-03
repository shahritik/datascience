import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 9 - Random Forest Regression')

dataset = pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values

#random forest for regression trees
#Step-1 take k data points from the dataset
#Step-2 build a decision tree for the k-datapoints
#Step-3 Choose n decision trees you want to build
#Step-4 predict the value of y using all the decision trees and take average

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state =0)
regressor.fit(x,y)

y_pred = regressor.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y,color ='red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Salary vs Level (Random Forest Regressor)')
plt.xlabel('Levels')
plt.ylabel('Salary')