# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/3. Regression/Support Vector Regression(SVR)/Position_Salaries.csv')
#print(dataset)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
#print(X)
#print(y)
y = y.reshape(len(y),1)
#print(y)

# Feature Scaling
#The Level and salary column has large scale difference hence we apply Feature scaling here.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#print(X)
#print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
#kernel can be of different types depending upon linear,non-linear,polynomial 
#or gaussian svr since our problem is non-linear we use "rdf" kernel
regressor.fit(X, y)#fit the dataset

# Predicting a new result
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
#print(y_pred)
#sc_X.transform-->since in feature scaling our dataset is scaled(transform) in some values range hence it is also require to scale the test value
#in the arguments of transform it requires a matrix(1x1 dimension) and not an array(vector)
#sc_y.inverse_transform-->converts or scale or transform back to the original scale of the salary

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()