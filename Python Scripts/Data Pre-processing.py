#Data Pre-processing

#Importing the libraries

#numpy includes all the mathematical tools
import numpy as np

#to plot the charts
import matplotlib.pyplot as plt

#import and manage datasets 
import pandas as pd

#setting the working directory
import os

os.getcwd()

os.chdir('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------')

#importing the dataset
dataset= pd.read_csv('Data.csv')

#extracting the independent variables
X = dataset.iloc[:,:-1].values

#extracting the dependent variable 
Y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#working with categorical variables
#encoding the cateforical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

#now the countries in the first column are encoded with 0, 1, 2
#but actually the countries are not compareable i.e. we will encode it with dummy encoding
#for this we create onehotencoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X=onehotencoder.fit_transform(X).toarray()

#creating labelencoder for Y i.e. purchased
labelencoder_Y = LabelEncoder()
Y=labelencoder_X.fit_transform(Y)

#splitting the dataset into test and training
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state = 0)

#dont learn the training set by hard rather understand the logic so as to predict the test set results

#feature scaling
#why is scaling important: most machine learning models are based on eucledian distance
#observation with wider range of values will dominate eucledian value
#2 types of feature scaling, standardization and normalization
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
X_train= sc_x.fit_transform(X_train)
X_test= sc_x.transform(X_test)

#even if the model is not based on eucledian distances we have to scale the values (eg. Decision Trees)























