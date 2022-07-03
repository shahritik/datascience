import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression')

dataset = pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values

#decision trees are of 2 types: classification trees and regression trees
#regression trees are more complex
#the algorithm is caplable of splitting the dataset on basis of information
#the final splits are called as terminal leaves
#very powerful model for 2-D

#implementing the decesion tree algorithm 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state =0)
regressor.fit(x,y)

#predict a new result
y_pred = regressor.predict([[6.5]])

#visualizing the result
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Salary vs Levels (Decision Tree Regressor)')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

