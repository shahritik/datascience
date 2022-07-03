#CNN
#image recognition
#computer processing image
#identifying the label of an image
#identifying the facial expressions
#Step-1: Convolution
#we input an image, it has pixels assigned to it
#the image is then passed to a feature detector
#feature matrix is a 2D matrix, the size of which can vary, the most common is 3*3
#feature detector is also called kernel or filter
#we multiply the input image and the feature detector by putting the filter on input image
#we make another matrix storing the total number of 1's that match in image and filter
#the image(matrix) on the right is called a feature map
#basically creating a feature map makes the image smaller to make it is easier to process it
#this might be a case that we loose quality if we reduce the size of image
#but it is not the case, whenever we see an image, we don't see each pixel but the features
#we create multiple feature maps using different filters

#Step 1b: ReLU layer
#images are highly non linear
#sometimes the feature map can be linear, in order to break the linearity we apply a recitfier function
#Step 2: Max Pooling
#if the feature in the image is distorted or loacated elsewhere, our CNN should have the flexibility to identify that feature and recognize the image
#we apply max pooling to feature map
#we create a pooled feature map using a 2*2 or any dimension matrix and then we record the maximum value in that
#the benefit of pooling is that we are reducing the size (helps in terms of processing), we only see the main features, we prevent overfitting 
#we are able to consider the facial or positional distortions of the image
#instead of max we can have sum, average, min pooling
#Step 3: Flattening
#Basically converting pooled feature map into a single column 
#we later input this column into artificial neural network

#Step-4: Full Connection
#add ann to cnn
#in cnn we will be using fully connected hidden layers
#there are multiple outputs
#we check the error: basically the cost function
#to minimize the error we use the back propogation and change the weights or the feature detector

#Softmax and Cross entropy
#we apply the softmax function in the final predicted probabilities to add them to 1
#cross entropy function is used instead of mean square method (cost function) to check the error or percentage accuracy
#why we use cross entropy over mean squared error
#if the error is very less and when we do back propogation to reduce it, in case of mean squared error it will not be able to detect
#in case of cross entropy since it has logarithm in it, it will detect even the tiniest of error and tell the weights in which direction to move
#cross entropy will help to NN to go to a better state

#importing the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)')

#building the CNN

#importing all the keras libraries needed
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#since videos are 3D with time we will use Convolution3D and MaxPooling3D

#initializing the CNN
classifier=Sequential()

#adding the different layers
#Step-1 : Convolution
#GPU can be used for having a very high processing power
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))
#if you are using any other backend apart from tensorflow we have to change the format of input_shape
#we will use the activation function to make sure that we do not have any negetive values in our feature map to make sure we get the non-linearity

#Step-2: Pooling Step
#reducing the size of feature map
#we use the max pooling and move with a stride of 2 i.e. skipping 2 columns
#after applying pooling the size of feature map is reduced by half if size is even
#if size of feature map is odd it is reduced by half+1

classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding another convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#we also have to do max pooling again

#Step-3: Flattening
#making a single vector which will be the input layer of ANN
#we cannot flatten the original image directly because we will only get information for each pixel and not for the pixels around it
#we create many pooled feature maps and flatten all of them into 1 single 2D array
#as each feature map correspnds to one specific feature of the image
#therefore each node of 1D array which contains a high number will represent a specific feature
#thus complete 1D array will represent all the feature of the image

classifier.add(Flatten())

#Step-4: Full Connection
#add the hidden layer
#number of nodes in the hidden layer must be around 100 for optimum but better if it is a power of 2
classifier.add(Dense(output_dim= 128, activation = 'relu'))

#add the output layer
#we use a sigmoid function because it is binary outcome: cat or dog
#if we had outcome for more than 2 categories we use the softmax activation function
classifier.add(Dense(output_dim= 1, activation = 'sigmoid'))

#compiling the CNN
#if the output is not binary and than we use categorical_crossentropy as loss function
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#fit the cnn to our images
#overfitting is getting great results on the training set but bad results in the test set
#therefore we have to do data pre processing

#image pre-processing or image augmentation
#we take some code from keras documentation

#first step is image data generator
#if we have less data to train it may generate correlation among itself causing overfiting
#to avoid it image data generator comes into play

#it will create many batches of our images by rotating, flipping and shifting the images
#image augmentation is a method to enrich our dataset and prevent overfiting

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, #rescale all the pixel values between 0 and 1 which are originally between 0 and 255
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

#increasing the test set accuracy
#here without parameter tuning we can either add another convolutional layer or another dense layer

#we add second convolution layer
#code changed above

#with 1 convolutional layer test set accuracy was 75% 
#with 2 convolutional layer test set accuracy was 82%
#one more way is to increase the number of pixels from 64, but use a GPU









