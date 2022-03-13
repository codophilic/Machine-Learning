# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#print(dataset)
X = dataset.iloc[:, 1:2].values #always a matrix(n*m) not a vector(array)
y = dataset.iloc[:, -1].values

#Polynomial Lineae Regression-->Y=b0+b1x1+b2x2^2+b3x3^3+........bnXn^n polynomial equation !! Then Why Linear?
#Linear beacause the coefficients are linear b0,b1,b2
#Non-linear coefficients are Y=b0+b1x1/(b0+b1)

#Here we will not split the dataset into test set and train set beacuse the  size of dataset (no. of rows) if we split we cannot get a accurate 
#prediction since datasets are not enough.

#CCOMPARISON WITH SIMPLE LINEAR REGRESSION V/S POLYNOMIAL LINEAR REGRESSION.

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression #----> SIMPLE LINEAR REGRESSION
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures #-----> POLYNOMIAL LINEAR REGRESSION 
poly_reg = PolynomialFeatures(degree = 4) #PolynomialFeature(degree=2(default)(it gives different types of curves depending on degree))
#degree=4 give perfect curve for our dataset 
X_poly = poly_reg.fit_transform(X)#It fit and transforms our matrix X into a new matrix with additional features(1,X,X^2,X^3,X^4) depending on the degree
lin_reg_2 = LinearRegression() #coeff. are linear 
lin_reg_2.fit(X_poly, y)#fitted our matrix into Linear Regression



# Predicting a new result with Linear Regression
#lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
#lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))




#COMPARE USING VISUALISAITON


# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'green')
#Here if we write lin_reg_2.predict(X)-->Here the lin_reg_2 is an object of linear regression class and not an object of polynomial regression class
#Here to add the polynomial feature "lin_reg_2.predict(X_poly)" but in Plot(X_new,....) a new matrix of X is there we have to take "poly_reg,fit"
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1) #levels
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

