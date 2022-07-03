#optimizing the sales in a grocery store

setwd('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori')

dataset=read.csv('Market_Basket_Optimisation.csv', header = FALSE)

#choose the association rule learning algorithm to see where to place the products in the store
#keeping milk and cereals close to each other have greater chances of putting the products in the same basket

#we will pre-process the data and make sparse matrix
library('arules')
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=100)

#training the apriori model for our dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

#visualizing the results
inspect(sort(rules, by = 'lift')[1:10])