# Grid Search

#For improving the model performance, a technique is called model selection.
#The model selection consist of choosing best parameters for your machine learning.
#There are two types of parameters:
    #1) The parameters model learns that is the parameters that were changed and found 
        #new optimal values by running the model.
    #2) the parameters which our chosed by ourself e.g kernel parameter..these parameters
        #are called hyper parameters
        #So there is still room for these parameters to find optimal values but since
        #these optimal values for parameters are not learned by the model so for this
        #we will use grid search.
#Before applying Grid Search we need to evauate our model
#Here we split the dataset into test set and train set and run or evaluate our model.
#after that we get some good accuracy for that particular dataset. When we apply a new dataset
#using previous dataset information our accuracy decreases and also ceates variance problem.
#So judging our model performance based on one dataset is not relevant.
#So to fix these variance problem we use K-Fold Cross validation technique.

#Problem with splitting the dataset into training & testing sets-->Now in both the sets
#we want to maximize our result or performance but there some datapoints we take out from
#training set and put in test set(splitting) we lose certain information will results 
#in incorrect prediction. e.g we take first 75% dataset as training set and use last 
#25% as test set, it is not gaurantee that this split we give best performance, maybe 
#there the last 75% training set gives best performance ?. 
#To over come such overlapping problem we use K-fold Cross Validation.

#What does K-fold Cross Validation do?
#1) It first divides the dataset into K-sets.
#   e.g K=3 so we get 3 parts of dataset 1 2 3 |4 5 6| 7 8 9-->3 folds
#2) Now each fold will be use as test set and remaining fold will be used has train set
    #e.g the first set (1 2 3) will be test set and (4...9) train set. So this model will
    #train and will give some performance or accuracy values
#3) Now again step 2 is repeated but now we will use different fold
    #e.g (4 5 6)-->as test set and (1 2 3 7 8 9)-->train set
#4) Now again step 2 but test set--->(7 8 9), train set-->(1 2 3 4 5 6)
#5) This was K-iterations depending on Folds value are followed
#6) The best accuracy model from the iteration is taken out.

#So trying different different cross-validation or K-folds we get to know 
#what can be the best & worst accuracy of our model K-Fold Cross Validation.     

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation #This part is evaluating our model.
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#cv=K
#accuracie-->array storing the accuracy of 10 models of iterations(K=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))#tells us variance as low as possible.

#'StratifiedKFold' class it will divide each of the cateogries in a uniform way
#e.g suppose there are 4 types of different flowers in Iris dataset, so the StratifiedKFold
#will split the data in such a way that atleast each cateogries flowers are present in
#the training set not like while splitting 3 categories are in training and there is no sample
#of 4th cateogry in training sets.


#below part is improving our model performance.
#The Grid Search answers which optimal parameters is good if 
#we use a particular classifer.e,g SVM-->all its internal parameters.
#Now to which model to choose
#(linearly seperable classifer(Linear SVM) or non-linerarly classifier(kerne SVM))? follow grid search.
#Now to code Grid search it is necessary to perform our model or fit our model
#since our one parameter of Grid search is 'estimator'.
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 
               'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 
               'kernel': ['rbf'], 
               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
              ]
#parameters--> it is a variable to find the optimal model of SVM.
#Here we created two dictonaries 1. Linear & 2. rb
#Now in SVC:
    #1) C-->its a parameter for regularization to prevent overfitting(should be 
            #not more than 10000 or it will go under-fitting)
            #so [1,10,100,1000] each value is try with kernel function and grid search 
            #will give best value.
    #2) gamma-->used for rbf(1/n_features--> so values <1 are used)
#Grid Search also evaluate K-Fold Cross Validation.(cv=10)
#n_jobs-->uses all processing power for faster performance on large dataset.
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
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
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()