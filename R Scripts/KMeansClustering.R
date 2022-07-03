#setting the working directory 
setwd('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering')

#importing the dataset
dataset = read.csv('Mall_Customers.csv')
x=dataset[,4:5]

#choosing the optimum number of clusters
#using the elbow method 
set.seed(6)
wcss <- vector()
for (i in 1:10) {
  wcss[i]<-sum(kmeans(x,i)$withinss)
}
plot(1:10, wcss, type = 'b',main = paste('Cluster of clients'), xlab = 'Number of Clusters', ylab = 'WCSS')

#applying kmeans to the mall dataset
set.seed(29)
kmeans <- kmeans(x, 5, iter.max = 300, nstart = 10)

#visualizing the clusters
library(cluster)
clusplot(x,kmeans$cluster,lines = 0,shade = TRUE, color = TRUE,labels = 2, plotchar = FALSE, 
         span = TRUE, main = paste('Cluster of clients'), xlab = 'Number of Clusters', ylab = 'WCSS')