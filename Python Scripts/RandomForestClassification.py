#random forest classification 
#ensemble learning 
#take multiple machine learning algorithm and create one bigger machine learning algorithm 
#random forest is basically making multiple decision trees or running the decision tree algorithm multiple times 
#Step 1: pick random K points from your dataset
#Step 2: build a decsion tree using those K datapoints 
#Step 3: Choose the number of trees you want to build and repeat steps 1 and 2
#Step 4: For the new datapoint: predict using the N-trees in which category the datapoints fall, and decide on the majority of votes

#random forest is like an army of decision trees each one making a different prediction

#getting the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 


os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 20 - Random Forest Classification')

dataset= pd.read_csv('Social_Network_Ads.csv')

#independent variables
x=dataset.iloc[:,2:4].values
#dependent variables
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0)

#Feature Scaling
#random forest too does not depend on eucledian distances but we have to apply scaling as the graph is based on 0.01 resolution
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#fitting classifier for the training set results
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

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
plt.title('Random Forest Classifier (Training set)')
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
plt.title('Random Forest Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#basically decision tree and random forest do overfitting in this kind of dataset and therefore the naive bayes and Kernel SVM are the besr algorithms are the best suited for this 

