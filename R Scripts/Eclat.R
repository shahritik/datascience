#Eclat
#this only includes the support of subsets 
#i.e. how likely is it for a subset to occur in a set of sequences 
#we set up a minimum support and than take out pairs having support greater than the min support
#sort the results in descending order of their support
#we will get the products frquently bought together
#no confidence required

setwd('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 29 - Eclat')

dataset=read.csv('Market_Basket_Optimisation.csv')

library('arules')
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=100)

#training the eclat model for our dataset
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

#visualizing the results
inspect(sort(rules, by = 'support')[1:10])
