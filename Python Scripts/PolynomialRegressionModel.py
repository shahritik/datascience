import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression')

dataset= pd.read_csv('Position_Salaries.csv')

#form = b0 + b1x1 + b2x1^2
#it is only dependent on one variable or predictor i.e. x1
#it is called linear because it is a function it is regards with the coefficient

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fit linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#fit polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
#x_poly has 3 columns, 1st for constant, 2nd linear, 3rd squared term

#creating linear regression
lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)

#visualizing the linear regression model
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color= 'blue')
plt.title('Salary vs Level')
plt.xlab('Level')
plt.ylab('Salary')
plt.show()

#visualizing the polynomial regression model
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color= 'blue')
plt.title('Salary vs Level')
plt.xlab('Level')
plt.ylab('Salary')
plt.show()

#predicting a new result with linear regression model
lin_reg.predict([[6.5]])

#predicting a new result with polynomial regression model
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))