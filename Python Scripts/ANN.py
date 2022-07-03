#DeepLearning
#ANN
#for neural networks and deep learning you need a lot of data and strong processing
#in future, we are planning to use DNA for storage
#deep learning is basically mimicking human brain
#nuerons by themselves are useless
#it has a center part which is called the body of the neuron, the tail is called an axon
#a lot of branches coming out from the head are called dendrites
#a lot of neurons together can build magic
#dendrites are the receiver of the signal and axon are the transmitter of the signal
#dendrites are connected to the axons of another neurons
#the signal from the dendrites is passed to the body of the neuron through synapses
#the input variables to the dendrites are basically all the independent variables
#you need to standardize the input variables
#sometimes instead of standardizing the variables you have to normalize them depending on the scenario
#the output values can be continuous or binary or categorical 
#we assing weight to synapses, weights are how ANN learn
#basically the weight decides to what extent the signal is passed
#what happens inside the neuron
#first: it takes the weighted sum of all the values 
#to the weighted sum it applies an activation function 
#after this the neuron decides to pass the signal or not or what signal to pass on
#what is activation function
#there are total 4 different types of activation function
#threshold function
#on x-axis we have weighted sum of inputs
#on y-axis we have 0 and 1 as the value
#if x<0 y=0, if x>=0 y=1
#Sigmoid Function= 1/1+e^-x (x is the value of the weighted sum)
#it is a smooth increasing function, important when we want to predict probabilties 
#Rectifier function
#till x<0 y=0, x>0 it increases linearly 
#it is one of the most used functions in ANN
#Hyperbolic tangent function
#very similar to the sigmoid function
#but the values of y go below 0 for x<0
#range of values -1 to 1
#for this tutorial in the hidden layer we apply the rectifier function, for the output layer we apply the sigmoid function
#how do NN works?
#the extra power that we have in neural network is due to the hidden layer present
#basically few random input values are picked up to form a combination i.e. a new independent variables
#these different combinations are assigned to different neurons and thus the combination these neurons make it powerful
#how NN learns
#programming is where you define the rules to reach to a certain output using an input
#while NN is where you have an input and output, the machine itself learn the rules to reach to that output
#a basic single layer of NN is called a preceptron 
#perceptron is something that can learn and adjust itself
#in order for the machine to learn we have to compare the output and the actual value
#the actual value is represented by y and output value by y^(y hat)
#the difference is calculated using the cost function = 1/2(y^-y)^2
#our goal is to minimize the cost function
#the cost value is sent back to the neuron and from there we can modify the weights
#1 epoch is when we go through the whole dataset and train the NN
#we calculate the predicted value of each row and comparing it with the actual value we calculate the cost function by summation
#then we update the weights 
#we update the weight for the input values, there is one weight corresponding to all the rows
#this whole process of going back and adjusting the weights is called backpropogation
#there are many different types of cost functions: research online lecture 224
#gradient descent 
#gradient descent is a method to minimize the cost
#one brute approach is that we assume a set of different weights and find the most optimum combination
#but then comes the curse of dimensionality 
#we cannot just assume weights and try their permutations
#we find the slope and see if it positive move to the left and of is is negetive move to the right
#it is called gradient descent because you are descending to the bottom or the minimum
#stochastic gradient descent
#the compulsory condition for gradient descent is that the cost function must be convex
#if it is not we might find a local minimum instead of a global minimum 
#in batch gradient descent you adjust the weights after you predict for all the rows
#in stochastic gradient descent you adjust the weight ater you predict every row
#stochastic gradient descent avoids the problem of capturing the local minimum
#there is also a mini gradient descent where you run batches of rows at a time
#backpropogation adjusts all the weight at the same time
#Step-1: randomly initialize the weights somwhere near to 0
#Step-2: Input the observation
#Step-3: Forward propogation
#Step-4: Compare the results and calculate the error
#Step-5: Back propogation (update the weights as per how much they are responsible for error, this is calculated by learning rate)
#Step-6: repeat steps 1 to 5 and update weight after each observation, this is called reinforcement learning, also stochastic gradient descent
#Step-7: When whole training set is passed through ANN it is called one epoch

#importing the dataset
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 

os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)')

df_ann=pd.read_csv('Churn_Modelling.csv')

#first we have installed keras 
#we solve a classification problem 
#this is the first branch of deep learning
#second is CNN (Convolutional Neural Networks) used for computer vision

#pre processing the data
x_ann=df_ann.iloc[:,3:13].values
y_ann=df_ann.iloc[:,13].values

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1=LabelEncoder()
x_ann[:,1]=labelencoder_x_1.fit_transform(x_ann[:,1])
labelencoder_x_2=LabelEncoder()
x_ann[:,2]=labelencoder_x_2.fit_transform(x_ann[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x_ann=onehotencoder.fit_transform(x_ann).toarray()

#remove one column to avoid the dummy variable trap
x_ann=x_ann[:,1:]

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_ann_train,x_ann_test,y_ann_train,y_ann_test=train_test_split(x_ann,y_ann,test_size=0.20,random_state=0)

#apply feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_ann_train=sc.fit_transform(x_ann_train)
x_ann_test=sc.fit_transform(x_ann_test)

#importing the keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initalizing the ANN
classifier_ann=Sequential()

#adding the input layer and the first hidden layer for ann
classifier_ann.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#adding the second hidden layer
classifier_ann.add(Dense(output_dim=6,init='uniform',activation='relu'))

#adding the output layer
classifier_ann.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the complete ANN
classifier_ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to the training set
classifier_ann.fit(x_ann_train,y_ann_train,batch_size=10,nb_epoch=100)

#predicting the test set results
y_ann_pred=classifier_ann.predict(x_ann_test)
y_ann_pred=(y_ann_pred>0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_ann_test,y_ann_pred)