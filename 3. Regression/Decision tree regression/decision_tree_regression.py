# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
#Here the default criteria(argument used) is "mse"-->Mean Squared error
#the "mse" reduce the variance(variation of values) and applies mean on each node
#on the tree and minimised the error. 
#for more arguments use GOOOGLE.
regressor.fit(X, y)


# Predicting a new result
y_pred=regressor.predict([[6.5]])
print(y_pred)


from sklearn.tree import export_graphviz
export_graphviz(regressor,out_file='tree.dot',feature_names=['decision'])
#it gives a tree structure of the decision tree applied on the DATASET
#in a word file format


#If we plot the graph without using higher resolution we see that it gives a 
#polynomial like curve which gives poor prediction and not required since in the datasets the difference of interval
#is larger (X values) hence to get a desired plot we use interval having small differances
#or higher resolution. 
#A decision tree has a non-continuous plot 
#Hence a decision tree does not fit good for a 1-dimensional model but it will be very good
#for higher dimensional model

 
# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()