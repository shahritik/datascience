setwd('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)')

dataset = read.csv('Position_Salaries.csv')

dataset = dataset[,2:3]

#fitting the SVR model
library('e1071')

#type is important to check if you are making a SVN model or SVR model
#SVN model is used for Classification
regressor = svm(formula = Salary ~ ., data = dataset, type = 'eps-regression')

#visualization the regression results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             color = 'red') + 
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), color = 'blue') +
  ggtitle("Salary vs Level") +
  xlab("Level") + 
  ylab("Salary")

y_pred = predict(regressor, data.frame(Level = 6.5))