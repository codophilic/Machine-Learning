# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Vector-->it is an array with elements no dimension e.g [6.5 4.5 5.5 ]-->(3,)
#matrix-->[[2,3],[3,4]]-->(2,2)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #.iloc[No. of rows, No. of column] values-> take all the values in the dataset
#print(dataset)
#print(dataset.iloc[:,0]) #[display number of rows e.g 2:3 , display number of columns(start of column with index 0) e.g :-->prints complete dataFrame]
# If the there is only 2 column then the output of the dataset.iloc[:,:-1]==dataset.iloc[:,0] but the way of packing the data is different in both the case
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
y = dataset.iloc[:, -1].values
print(X)
print(y) #dependent variable 


# Taking care of missing data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#the data missing in the dataset is written has "NaN", strategy means to find the data in a  method way which here is "mean", axis=0(column) axis=1(row)
imputer.fit(X[:, 1:3])
#Here  we are specifying to fit the data in the given column not on the whole dataset, X[No. of rows, No. of column]
X[:, 1:3] = imputer.transform(X[:, 1:3])
#put the fitted data back to the original dataset of column.
print(X)



# Encoding categorical data ---> The column having string in different names(categories),since ML models are based on mathematical equations and numbers
#that's why we encode the text or different strings.

#The column 0 in X and Y column has a label(string)
#The column 0 in X--->Germany,Spain,France
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#X[:,0] = le.fit_transform(X[:,0])
#print("Problem")
#print(X)
#[[0 44.0 72000.0]
# [2 27.0 48000.0]
# [1 30.0 54000.0]
# [2 38.0 61000.0]
# [1 40.0 63777.77777777778]
# [0 35.0 58000.0]
# [2 38.77777777777778 52000.0]
# [0 48.0 79000.0]
# [1 50.0 83000.0]
# [0 37.0 67000.0]]


# Here The City has assigned(0,1,2) 3 cateogries which is like having a high precedence or most(as per ML understanding this number) which is not actually.
#E.g Germanty is Greater than France
#To overcome this we use dummy variables concept
#Country    France Germany Spain
#France        1     0       0     
#Germany       0     1       0
#Spain         0     0       1
# Here we require 3 dummy varaibles nut we use only two because based on the two we can get infomation about the third column
#So if there are 'n' dummy varaibles use only 'n-1'
#if we consider all the dummy variables then there will be a Dummy variable trap
#suppose there are two dummy variable Y=b0+b1D1+b2D2+rest--->b0=constant of equation,  D1,D2-->dummy varaibles 
#now in thw encoder if D1=0 and D2=1 OR D1=1 and D2=0,then the equation becomes Y=b0+(b1+b2)+rest 
#Here b1+b2 becomes another constant like b0 which creates a problem of Multicollinearity problem and Regression will not able to run.
#So to avoid it either use 'n-1' dummy variables OR do not use the constant.

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#The constructor of the ColumnTransformer takes few arguments but required arguments are "transformers" and "remainder"
#transformer takes a list has parameter containing [name of the taransformer(setting parameters and 
# searching purpose),transformer(estimator)=OneHotEncoder(),list of column number] , remainder(this will tell the transformer what to do with other column)
#default is "drop" which will return the transform column or "passthrough" does not change anything and return it.
X = np.array(ct.fit_transform(X)) #it fit the transformer into dataset and return the dataset
print(X)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
#explanation refer Linear Regression
from sklearn.model_selection import train_test_split
#In the latest version cross_validation is removed and in-replace use 
#this above split module.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
#A higher value number must not dominate a lower value number so Feature scaling makes them to scale down to small and same values in range
#E.g experience(x1) workpoints(x2)
#        1          10000
#        2          30000
#        3          50000
#and so on, here while implementing the Y=b0+b1x1+b2x2 x2 values range are higher than x1 values range and therefore the model will depend on x1 which
#will mot be able to give accurate result.
#the same is applicable to the dependent variable(Y)
from sklearn.preprocessing import StandardScaler
#z=(x-u)/s-->x=datasets elements,u=mean,s=standard deviation
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])#fit and transform the training set, transform is like conversion or scaling and 
#fit is like fitting to dataset.
X_test[:, 3:] = sc.transform(X_test[:, 3:])#transform the test sets using fitted model 
print(X_train)
print(X_test)