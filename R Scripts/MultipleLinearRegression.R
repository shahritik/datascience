#set the working directory
setwd('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression')

#importing the dataset
dataset = read.csv('50_Startups.csv')

#encoding the categorical variables
dataset$State=factor(dataset$State, levels = c('New York', 'California','Florida'), labels = c(1,2,3))

#splitting the data in test and training set 
library(caTools)
set.seed(0)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set=subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fitting the multiple linear regression model to training set
#R autmatically removed the dummy variable and cosidered the constant
regressor = lm(formula = Profit ~ ., data = training_set)

#predicting the test set results
Y_pred = predict(regressor, newdata = test_set)

#backward elimination in R (building the optimal model)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)
summary(regressor)

#remove the State 2 and State 3 variable 
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regressor)

#remove the Administration variable
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = dataset)
summary(regressor)

#remove the Marketing Spend variable
regressor = lm(formula = Profit ~ R.D.Spend, data = dataset)
summary(regressor)

#profit is directly related to R.D.Spend (highly statistically significant)



