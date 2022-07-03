#importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#set the working directory
os.chdir('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression')

#import the dataset
dataset = pd.read_csv('50_Startups.csv')

#you should not include all the dummy variables in your model
#0 in column specifies that the next dummy variable has a value 1 in a 2 dummy variable case: dummy variable trap
#the constant includes the effect of eliminated dummy variables
#see the intutuon video of multiple linear regression model
#soo whenever you build a model using categorical variables always omit one dummy variable
#if you have 2 sets of dummy variables apply this rule to each set
#you always have to decide which independent variables to keep in and out if the model
#if there's a lot of variables for the model it's not good
#you also explain each variable
#therefore keep only the most important and right variables
#5 models for building models
#Step-wise regression: Backward Elimination, Forward Selection, Bi-directional elimination
#All-in (include all the variables (usually avoid)), Score comparison

#Backward-Elimination
#Step-1 Choose a Significance value- 0.05
#Step-2 Build a model using all the independent variables
#Step-3 Consider the predictor with highest P-value, if P>SL go to step-4, else finalize the model
#Step-4 Remove the predictor
#Step-5 Fit the model without this variable 

#Forward Selection
#Step-1 Choose a significance value- 0.05
#Step-2 Fit all simple linear regression models y~xn. Select the one with the lowest P-value
#Step-3 To this simple linear model add the other predictors one by one and choose the model in which the 
#newly added variable has the lowest P-value
#Step-4 After cosidering the predictor with lowest P-value and if P<SL move to step-3

#Bi-directional elimination: Step-wise elimination
#Step-1 Select a Significance level to enter and to stay- 0.05
#Step-2 Perform all the steps of forward selection
#Step-3 Perform all the steps of backward elimination
#Step-4 No new variables can enter and no old variables can exit

#All possible models, and choose the one that gives the best prediction efficiency
#but very time consuming and resource consuming

#separating the dependent and independent variables
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the dummmy variable trap
X = X[:,1:]

#splitting the test and training
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state = 0)

#the library will take care of the feature scaling

#Implementing Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

#predicting the test set results
Y_pred= regressor.predict(X_test)

#building optimal model using backward elimination
import statsmodels.api as sm
#adding a column of 1 i.e. b0x0, x0=1 is not considered by statsmodels 
X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis =1)
#creating the matrix consisting only most significant variables
X_opt= X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog= X_opt).fit()

regressor_OLS.summary()

#remove the x2 as the P-value is (highest) greater than the significant value (0.05)
X_opt= X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog= X_opt).fit()

regressor_OLS.summary()

#remove x1 now
X_opt= X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog= X_opt).fit()

regressor_OLS.summary()

#remove x4 now
X_opt= X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=Y, exog= X_opt).fit()

regressor_OLS.summary()

#remove x5 now
X_opt= X[:,[0,3]]
regressor_OLS = sm.OLS(endog=Y, exog= X_opt).fit()

regressor_OLS.summary()

#check the tutorial for automatic implementation of backward elimination


















