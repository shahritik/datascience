#Decision Tree Classification
#2 types of carts: Classification trees and Regression trees
#Classification trees: classifiying the data between 2 categories of 3 categories
#regression trees: calculation of exact values 

#Function of Decision Tree is to split the data into groups so that each split has maximum number of similar categories 
#the split process is based on entropys

#after the final split the obeservations are called as leaves 

#decision trees can go very long, therefore sometimes we may leave the split data in between and classify a new observation based on probability

#getting the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 


os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 19 - Decision Tree Classification')

dataset= pd.read_csv('Social_Network_Ads.csv')

#independent variables
x=dataset.iloc[:,2:4].values
#dependent variables
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0)

#Feature Scaling
#we don't have to apply feature scaling as decision tree is not based on eucledian distance 
#but when we plot the graph of training and test set results, we use a resoultion of 0.01
#the code will execute much faster if the values are scaled
#if we want to plot the decision tree as a real tree itself then we can remove the scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#fitting classifier for the training set results
from sklearn.tree import DecisionTreeClassifier
#basically there are 2 criterion for classifer: gini and entropy 
#maximum algorithms are based on entropy: for example the NLP
#the more homogeneous the leaf nodes are the lower is the entropy, if the entropy is 0 that means it is a prefect split
#basically we calculate the information gain i.e. the difference between the initial entropy and the final entropy 
#higher the information gain the better the the decision tree classifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state =0)
classifier.fit(x_train,y_train)


#predicting the test set results
y_pred = classifier.predict(x_test)

#making the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#visualizing the training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#visualizing the test set results
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

