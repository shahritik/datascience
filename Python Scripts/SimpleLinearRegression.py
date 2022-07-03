#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#set the working directory
os.chdir('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression')

#Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Splitting the dataset
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=1/3, random_state = 0)

#Simple Linear Regression
#get the line of best fit y(dependent variable) = b0(constant) + b1(coefficient)x(independent variable)
#use the method of least squares
#draws a number a lines and everytime calculates the square distance with the datapoints
#line with least squared value is used as the regression model
#for simple linear regression we dont have to apply feature scaling, the library will take care of scaling
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set results
Y_pred = regressor.predict(X_test)

#visualizing the training set results 
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlab('Years of Experience')
plt.ylab('Salary')
plt.show()

#visualizing the test set results 
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlab('Years of Experience')
plt.ylab('Salary')
plt.show()

