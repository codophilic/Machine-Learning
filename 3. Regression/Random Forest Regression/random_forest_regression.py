# Random Forest Regression
#non-continous model
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
#n_estimators-->no. of trees in the forest and rest same argument for
#decision tree
#if we add more trees in the model it does not mean that we will get a lot more 
#steps in the stairs in then plot, the more the number of trees the more the average
#of the different predictions made by the trees is converging to the same average.
#it will converge some steps.

#n=100 we get around 150K as o/p
#n=300 we get around 160K as o/p(perfect!!!!!! prediction) 
regressor.fit(X, y)
#fit the regressor into our dataset

# Predicting a new result
print(regressor.predict([[6.5]]))

#without higher resolution we get a poor prediction curve
# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()