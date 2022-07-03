#there are 2 types of HC: agglomerative and divisive
#agglomerative is the bottom up approach
#Step-1: consider each datapoint as an individual cluster i.e. n clusters for n points 
#Step-2: take the 2 closest datapoints and bind them into 1 cluster i.e. N-1 clusters now
#Step-3: take the 2 clusters and combine them into 1 cluster i.e. N-2 clusters
#Step-4: repeat step 3 until there is only one cluster left
#to calculate the distance between 2 clusters there can be many methods and it is one of the most crucial steps for the HC algoithm
#measuring the distance between the closest points, farthest points, taking the average distance(combination of distance between all the points), distance between the centroids
#benefit of HC and its difference from kmeans clsutering algorithm 
#we create a dendogram 
#dendrogram is like a memory of the HC algorithm 
#on the x axis plot all the points and on the y-axis plot the eucledian distance
#join the clusters closest to each other by a horizontal line and height equal to the eucledian distance 
#while making a horizontal line between clusters choose the centroid of the 2 clusters 
#using this created dendrogram we enhance and compile our HC
#we have to decide a threshold for dissimilarity i.e. the clusters that are formed should not differ more than this threshold value
#basically we compare the threshold with eucledian distance and decide the number of clusters
#selecting the optimum number of clusters using the dendrogram
#select the longest vertical line which does not cross any extended horizontal line
#the length of longest vertical line is considered to be the threshold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 25 - Hierarchical Clustering')

dataset = pd.read_csv('Mall_Customers.csv')
x= dataset.iloc[:,[3,4]].values

#using the dendrogram to find the optimal number of clusters
#we use scipy library which is used for HC and building the dendrograms
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
#in kmeans we minimize the wcss between the clusters, here we use the ward method which is used to minimize the variance between each clusters
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distance')
plt.show()
#setting the threshold values as the largest distance which crosses no horizontal line we get optimum clusters as 5

#fit the HC to the dataset
#the most common is the agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity ='euclidean', linkage ='ward')
y_hc=hc.fit_predict(x)

#visualizing the results
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100, c='red',label = 'Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100, c='blue',label = 'Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100, c='green',label = 'Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100, c='cyan',label = 'Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100, c='magenta',label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300,c='yellow',label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

