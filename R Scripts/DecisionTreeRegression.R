setwd('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression')

dataset = read.csv('Position_Salaries.csv')

dataset = dataset[,2:3]

#making decision tree model
library(rpart)
regressor = rpart(formula = Salary ~ ., data = dataset, control = rpart.control(minsplit = 1))


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