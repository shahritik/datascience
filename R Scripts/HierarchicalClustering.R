setwd('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 25 - Hierarchical Clustering')

dataset = read.csv('Mall_Customers.csv')
x=dataset[,4:5]

#using the dendrogram to find the optimum number of clusters
dendrogram = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
plot(dendrogram, main = paste('Dendrogram'), xlab = 'Customers', ylab = 'Euclidean Distance')

#the optimum number of clusters is 5 

#fitting the HC for the dataset using 5 clusters
hc = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
y_hc=cutree(hc, 5)

#visualizing the results
library(cluster)
clusplot(x,y_hc,lines = 0,shade = TRUE, color = TRUE,labels = 2, plotchar = FALSE, 
         span = TRUE, main = paste('Cluster of clients'), xlab = 'Number of Clusters', ylab = 'WCSS')