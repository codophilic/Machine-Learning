import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Linear Regression 
# h(x)=coeff0+coeff1*x 
# h(x)=Dependent variable, x=Independent variable, coeff0=intercept of line , coeff1=Slope of Line
# Salary=coeff0+coeff1*EXPERIENCE

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#print(dataset)
#print(dataset.iloc[:,0]) #[display number of rows e.g 2:3 , display number of columns(start of column with index 0) e.g :-->prints complete dataFrame]
x = dataset.iloc[:,:-1].values
# Here the output of the dataset.iloc[:,:-1]==dataset.iloc[:,0] but the way of packing the data is different in both the case
# In dataset.iloc[:,:-1]--> we pack data per row wise(Nx1) hence the output is like 
# [[ 1.1]
 #[ 1.3]
 #[ 1.5]
 #[ 2. ]
 #[ 2.2]
 #[ 2.9]
 #[ 3. ]
 #[ 3.2]
 #[ 3.2]
 #[ 3.7]
 #[ 3.9]
 #[ 4. ]
 #[ 4. ]
 #[ 4.1]
 #[ 4.5]
 #[ 4.9]
 #[ 5.1]
 #[ 5.3]
 #[ 5.9]
 #[ 6. ]
 #[ 6.8]
 #[ 7.1]
 #[ 7.9]
 #[ 8.2]
 #[ 8.7]
 #[ 9. ]
 #[ 9.5]
 #[ 9.6]
 #[10.3]
 #[10.5]]

#In dataset.iloc[:,0]-->we pack the whole column which gets converted into a single list or array(1XN)
#[ 39343.  46205.  37731.  43525.  39891.  56642.  60150.  54445.  64445.
#  57189.  63218.  55794.  56957.  57081.  61111.  67938.  66029.  83088.
#  81363.  93940.  91738.  98273. 101302. 113812. 109431. 105582. 116969.
# 112635. 122391. 121872.] 

#print(x)
#print()
y = dataset.iloc[:, 1].values
#print(y)



# Splitting the dataset into the Training set and Test set
#Selecting 20 observations for training set and 10 observation for test sets that's why 'test_size=1/3' 30*1/3
#pre-process the data 

from sklearn.model_selection import train_test_split
#model_selection is a method for setting a blueprint to analyze data and then using it to measure new data. 
#Selecting a proper model allows to give accurate result when making predictions.
#To do that, you need to train the model by using specific dataset then test the model againts another dataset
#if one dataset is there then the dataset needs to be split
#'train_test_split'--> is a function for splitting the data araays into two subsets(training,testing) and it will make random partitions for the subsets.
#train_test_split(X,Y,train_size=0.*,test_size=0.*,random_state=*)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#The model will learn the correlations based on the training sets (x_train,y_train) so that it can predict the salary based on the informations.
#Once the model is trained we can check its power of predictions on the test sets made and then we can compare the predicted salary and actual salary(x_test,y_test).

# Fitting Linear Regression to the Training set. The model_selection module selects a appropriate training sets and now we fit a linear regressor to the training 
# sets
from sklearn.linear_model import LinearRegression
#(object)-->regressor of LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)#Fits the Linear Regression or fit method.
#.fits(independent variable, dependent variables)-->method
#we created a machine name regressor and learned the machine on the training sets to understand the correlations between experience and salary.


# Predicting the Test set results
y_pred = regressor.predict(x_test)
#print(y_pred)

#Difference between the predicted salary and the actual salary
#print("y_test  y_pred")
#for i in range(len(y_pred)):
#    print(int(y_test[i]),int(y_pred[i]))

#checks the accuracy of training sets
plt.scatter(x_train, y_train, color = 'red')# shows the dot scatters on the plot
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#checks the accuracy of test sets
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
#Here if we write 'plt.plot(x_train, regressor.predict(x_train), color = 'blue')' we still gets the same result because the model gives a fix lines after 
# the training only in the x_train there will be more scatter plot
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualization of the test set
plt.scatter(x_test, y_test, color = 'red')
plt.scatter(x_test, y_pred, color = 'blue')
plt.plot(x_test, regressor.predict(x_test), color = 'green')
plt.legend(["Linear Regression line","Actual values","predicted values"])
plt.title("Prediction of test sets")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

y_pred=regressor.predict(x_train)
#Visualization of the train set
plt.scatter(x_train, y_train, color = 'red')
plt.scatter(x_train, y_pred, color = 'blue')
plt.plot(x_test, regressor.predict(x_test), color = 'green')
plt.legend(["Linear Regression line","Actual values","predicted values"])
plt.title("Prediction of train sets")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



