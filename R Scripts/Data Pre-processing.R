Data=read.csv('Data.csv')

#filling the missing value with the mean
Data$Age=ifelse(is.na(Data$Age), ave(Data$Age, FUN = function(x) mean(x, na.rm = TRUE)), Data$Age)

Data$Salary=ifelse(is.na(Data$Salary), ave(Data$Salary, FUN = function(x) mean(x, na.rm = TRUE)), Data$Salary)

#working with categorical variables
Data$Country=factor(Data$Country, levels = c('France', 'Spain','Germany'), labels = c(1,2,3))

Data$Purchased=factor(Data$Purchased, levels = c('Yes', 'No'), labels = c(1,0))

#splitting the dataset into test and training 
#install.packages("caTools")
library(caTools)
set.seed(0)
split = sample.split(Data$Purchased, SplitRatio = 0.8)
training_set=subset(Data, split == TRUE)
test_set = subset(Data, split == FALSE)

#feature scaling
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])