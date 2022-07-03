setwd('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 18 - Naive Bayes')

dataset = read.csv('Social_Network_Ads.csv')

dataset = dataset[,3:5]

library('caTools')
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.75)
training_set=subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])