#k++ is implemented to find the exact position of the centroids 
#since the selection of the centroid is random their is a possibility of formation of different clusters than expected
#this is automatically dealt by k-mean algorithm with a k-means ++ algorithm 
#elbow method 
#elbow method is a trick to find out the most optimum number of clusters
#you start with one cluster and calculate the WCSS values i.e. the sum of square of distance between the centroid and each point
#as you increase the number of clusters the WCSS values decreases
#initially the values will decrease drastically and after some clusters the rate of decrease in the value of WCSS will decrease
#when we plot a WCSS curve i.e. WCSS value with the number of clusters, we get an elbow where the rate changes, this elbow is the optimum number of clusters

#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering')

#getting the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimal number of clusters
#for init in making the kmeans classification object we can use 'random' but we don't want to fall into the initialization trap
#therefore we use k-means++
from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
    kmeans=KMeans(n_clusters =i, init = 'k-means++', max_iter = 300, n_init=10, random_state =0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

#we get the elbow at 5 clusters, therefore the optimum number of clusters is 5
kmeans=KMeans(n_clusters =5, init = 'k-means++', max_iter = 300, n_init=10, random_state =0)
y_kmeans = kmeans.fit_predict(x)

#visualizing the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100, c='red',label = 'Cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100, c='blue',label = 'Cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100, c='green',label = 'Cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100, c='cyan',label = 'Cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100, c='magenta',label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300,c='yellow',label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
