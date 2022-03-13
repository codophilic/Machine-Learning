# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

#based on age and salary  we gonna decide wheather the user will buy SUV or 
#not(0 or 1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling(Age and salary values have high range)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
#The confusion matrix is used to evaluate wheather our predictions 'y-pred'
#compare to 'y-test' given by our model was accurate or not
#So in the confusion matrix there will be correct as well as incorrect predictions
#of our model and with the help of this maxtrix we will be able to know our 
#performance of our model.
from sklearn.metrics import confusion_matrix#--->its a function.
cm = confusion_matrix(y_test, y_pred)#(real values,predicted values)
print(cm)
#65+24=89 correct predictions
#8+3=11 incorrect predictions


# Visualising the Training set results
#0--did not buy SUV(dependent variable purchase is 0)
#1--purchased SUV(dependent varible purchase is 1)
#Analysis
#young users with low salary did not buy SUV car(red points)
#users who are older and with a high salary purchased SUV car(green points)
#older users with low salary also bought SUV cars.
#Some young users with high salary also bought SUV cars.

#Now the separation line is decision boundry(prediction boundry).
#The line is a straight line because our classifier is a Linear Classifier.
#Our independent is 2-D that's why we get a 2D plot.
#Now our classifier has made two prediction regions(red region and green region) using decision boundry.
#Now the red region is a region where the classifier catches(or predicts) all the users who
#did not buy the SUV and the green region is a region where the classifier(or predicts) catches 
#all the users who purchase the SUV.

#So the points are the real output(yes or no) of the training set.
#There are some green points in the red region which means there are some users
#who bought the SUV car inspite of low salary. Similarly in green region.
#These are some incorrect predictions because our classifier is a Linear Classifier
#and because our users are not linearly distributed. If the users where Linearly
#distributed then we will have the green points in green region and red points in red regions.

from matplotlib.colors import ListedColormap#--->its a class that colourized all the datapoints
#Now considering all the pixels point(x,y)(X1,X2) of the graph(not the dataset values) the classifier predicted 
#the result wheather its '0' or '1' and colourized it red or green.
#Since we use a step=0.01 we get a continous region in our plot.

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#Preparing the grid using pixel points
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#contourf--> colouring purpose of points
plt.xlim(X1.min(), X1.max())#limits
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()