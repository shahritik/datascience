setwd('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 9 - Random Forest Regression')

dataset = read.csv('Position_Salaries.csv')

dataset = dataset[,2:3]

#making decision tree model
library('randomForest')
set.seed(1234)
regressor = randomForest(x = dataset[1], y=dataset$Salary, ntrees = 100)


library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             color = 'red') + 
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color = 'blue') +
  ggtitle("Salary vs Level") +
  xlab("Level") + 
  ylab("Salary")

y_pred = predict(regressor, data.frame(Level = 6.5))