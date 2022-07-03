import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

os.chdir('/Users/ritik.shah/Desktop/Important/Data Science/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)')

dataset = pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values

#SVR does not include the feature scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
sc_y=StandardScaler()
x= sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
#choose the kernel, i.e. you want linear svr or polynomial or rbf svr
#poly and rbf both work for the polynomial problem
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

#visualizing the svr result 
#x_grid = np.arange(min(x), max(x), 0.1)
#x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color= 'blue')
plt.title('Salary vs Level')
plt.xlab('Level')
plt.ylab('Salary')
plt.show()

#predict the values
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))