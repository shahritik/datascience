#set the working directory
setwd('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression')

#import the dataset
dataset = read.csv('Salary_Data.csv')

#splitting the dataset in test and training
library(caTools)
set.seed(0)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set=subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#linear regression model
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#predicting the values
Y_pred= predict(regressor, newdata = test_set)

#visualization the training set results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), color = 'blue') +
  ggtitle("Salary vs YearsExperience (Training Set)") +
  xlab("Years of Experience") + 
  ylab("Salary")

#visualization the test set results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), color = 'blue') +
  ggtitle("Salary vs YearsExperience (Training Set)") +
  xlab("Years of Experience") + 
  ylab("Salary")





