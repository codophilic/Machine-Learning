#BigMart Sales Prediction

#The data scientists at BigMart have collected 2013 sales data for 1559 products across
#100 stores in different citie. 

#AIM--:To build a predictive model and find out the sales of each product at a particular
#      store(outlet). The dataset consist of properties of products and stores which play a 
#      key role in increasing sales. 

#The unique ID of each item, it starts with either FD, DR or NC. 
#If you see the categories, these look like being Food, Drinks 
#and Non-Consumables.

#Dataset information:
    ######--PRODUCT--######
    #1) Item_Identifier: Unique Product ID
    #2) Item_Weight: Weight of product
    #3) Item_Fat_Content: Whether the product is low fat or regular.
    #4) Item_Visibility: The % of total area allocated for that particular product
    #5) Item_Type: Types of item or categories
    #6) Item_MRP: price of products
    
    ######--OUTLET/STORE--######
    #1) Outlet_Identifier: Unique Store ID
    #2) Outlet_Establishment_Year: The year in which store was established
    #3) Outlet_Size: The Size of the store in terms of ground Area.
    #4) Outlet_Location_Type: Location of store
    #5) Outlet_Type: whether the outlet is a grocery store or a SuperMarket.
    #6) Item_Outlet_Sales: Sales of the Product in the particular store.

#DEPENDDENT VARIABLE/OUTCOME VARIABLE/ Y--> Item_Outlet_Sales

#INDEPENDENT VARIABLE/X-->:
    ##########----PRODUCT----#########
    #1) Item_Identifier: Used for ID Variable for product.
    #2) Item_Fat_Content: Low Fat items are generally used.
    #3) Item_Visibility: Display Area in our hypothesis.
    #4) Item_Type: Types of item bought so its used in our hypothesis.
    
    ##########----STORE----#########
    #5) Outlet_Identifier: Used for ID variable of outlet.
    #6) Outlet_Location_Type
    #7) Outlet_Size
    #8) Outlet_Type


import pandas as pd

train_data=pd.read_csv('Train.csv')
test_data=pd.read_csv('Test.csv') 
#'test_data' does not have Item_Outlet_Sales which we need to predict.

#Joining the two dataset
#combine them into a dataframe ‘dataset’ with adding a ‘source’ column specifying 
#where each observation belongs.
train_data['source']='Train_dataset'
test_data['source']='Test_dataset'
dataset=pd.concat([train_data,test_data],ignore_index=True)
#'ignore_index=True' allows to overlap the common data of both train & test set
#into the new dataset 


#------1. ANALYSIS/DATA EXPLORATION------
import warnings
warnings.filterwarnings("ignore")

#This shows us our column index labels datatypes(dtypes) 
#Some column are of 'object' dtype and some are of 'float64' dtype
print("Data Information")
dataset.info() 

#Count total FD,DR and NC 
dataset['Item_Type_Combined'] = dataset['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
dataset['Item_Type_Combined'] = dataset['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
dataset['Item_Type_Combined'].value_counts()




#Counting all the unique categories in each colummn
count_unique=dataset.apply( lambda x: len(x.unique()))

#Now checking the frequency of each categories in each column
categorical_column1=[x for x in dataset.dtypes.index if dataset.dtypes[x]=='object']
#above line will give a list of index labels rows having dtype='object' (Item_Identifier,Outlet_Identifier etc..)
categorical_column2=[ x for x in categorical_column1 if x not in ['Item_Identifier','Outlet_Identifier']]
#above line will remove  unique ID column
for col in categorical_column2:
    print('\nFrequency of Categories for variable',col,'\n',dataset[col].value_counts())
    #display each column categories frequency.
    #'dataset['Item_Type'].value_counts()'--> displays each column categories frequency.
    
#Counting the missing values in each column
count_missing_values=dataset.apply( lambda x: sum(x.isnull()))
#Here 'Item_Outlet_Sales' is a target variable so its missing value are not considered.

#data describe(min value,  max value ,total value ...etc)
data_describe=dataset.describe()

#---------2. DATA CLEANING ----------

#Computing Missing values of 'Item_Weight' & 'Outlet_Size' using pandas

#1.---------------FINDING MISSING VALUES IN 'Item_Weight'

#2 ways to find missing values for Item_Weight(it is just to show the code eventhough 
#'Item_Weight' is not coonsidered)

#-------1>
Item_Avg_Weight=dataset.pivot_table(values='Item_Weight', index='Item_Identifier')
#The 'pivot_table' takes simple column-wise data input and groups/aggregrate  
#the entries into 2D table that provides a multidimensional summarization of data.
#Its like fixing one thing and apply some aggregration operations (default='mean') on its index.
#e.g
#     A     B     C--->dataFrame
# 0  J   Mastter  20
# 1  M  Graduate  30
# 2  M  Graduate  40
# 3  P    Master  50
#applying df.pivot_table(values='C',index='A',aggfunc='sum')
#OUTPUT
# A  C
# J  20
# M  70
# P  50
#Here we applied aggregation function as 'sum',So the 'A' index column having
#two values of 'M' and its corresponding 'C' values got summed.  
miss_bool=dataset['Item_Weight'].isnull()
#Creates a list of boolean values if the column value is null or not
dataset.loc[miss_bool,'Item_Weight']=dataset.loc[miss_bool,'Item_Identifier'].apply(lambda x: Item_Avg_Weight.at[x,'Item_Weight'])
#'loc'-->It ia unique method to retrieve rows from a DataFrame. This method takes
#only index labels and returns row or DataFrame if the index label exists in
#the caller dataFrame.(Its like accessing rows or particular cell of dataFrame)
#e.g (above)
#df.index=['R_1','R_2','R_3','R_4']--->rows assign by names 
#a=df.loc['R_2']---> Accessing row R_2
#a
# label      R_2
# A           M
# B    Graduate
# C          30

#-------2>
#codeline1: Item_Avg_Weight=dataset.groupby('Item_Identifier').mean()['Item_Weight']
#('groupby' operation involves one of the following operations on the original 
#object:
    #1. Splitting 2. Applying Function 3. Combining Results
#It allows yout to split the data for better computation analysis.
#e.g Finding out avg. age in a cities classifying or spliting on the basis of
#gender,age etc.
#e.g 
    # Name  account House
    # H       10   Mum
    # M       20   Mum
    # J       30   Mum
    # H       40   Kol
    # M       50  Pune
    # J       60    JK
    # H       70  Pune

#a=df.groupby('Name').sum()['account']
# Name account
# H    120
# J     90
# M     70
# Name: account, dtype: int64)
#codeline2: miss_bool=dataset['Item_Weight'].isnull()
#codeline3: dataset.loc[miss_bool,'Item_Weight']=dataset.loc[miss_bool,'Item_Identifier'].apply(lambda x: Item_Avg_Weight[x])

#2.------------FINDING MISSING VALUES IN 'Outlet_Size'

#------1>
#Each Outlet_Type has a particular Outlet_Size
#Outlet_Size_Mode=dataset.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode().iat[0]))
#('mode'-->The Value which appears most frequently in a dataset
#e.g 
#dataset(S=Small,M=Medium,L=Large)
#  Size Store No.
#     S    Store1
#     S    Store1   
#     L    Store2
#     L    Store2
#     M    Store3
#     M    Store3
#     M    Store2
#     S    Store1
#a=df.pivot_table(values='Size',columns='Store No.',aggfunc=(lambda x:x.mode().iat[0]))
#print(a)
#Store No. Store1 Store2 Store3
#Size           S      L      M

#Here in the there are 3 types of store and corresponding are its sizes.
#We see store2 has two types of size (L,L,M) so the highest frequency is L
#Therefore Store 2 size is considered L. 
#miss_bool_size=dataset['Outlet_Size'].isnull()
#dataset.loc[miss_bool_size,'Outlet_Size']=dataset.loc[miss_bool_size,'Outlet_Type'].apply(lambda x: Outlet_Size_Mode[x] )

#------2>(NA)

#Checking Null values
#print(sum(dataset['Item_Weight'].isnull()))
#print(sum(dataset['Outlet_Size'].isnull()))

#----------------4. FEATURE ENGINEERING------------

#1.-------Modify 'Item_Visiblity'
#We noticed that the minimum value here is 0, which makes no practical sense. 
#Lets consider it like missing information and impute it with mean visibility 
#of that product.
visibility_avg = dataset.pivot_table(values='Item_Visibility', index='Item_Identifier')
#deafult aggfunc is mean
#Impute 0 values with mean visibility of that product
miss_bool = (dataset['Item_Visibility'] == 0)
dataset.loc[miss_bool,'Item_Visibility'] = dataset.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.at[x,'Item_Visibility'])

#Products with higher visibility are likely to sell more. 
#But along with comparing products on absolute terms, we should look at 
#the visibility of the product in that particular store as compared to the 
#mean visibility of that product across all stores. 
#This will give some idea about how much importance was given to that product 
#in a store as compared to other stores. 
#We can use the ‘visibility_avg’ variable made above to achieve this.
dataset['Item_Visibility_MeanRatio'] = dataset.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)
#
# Item_Identifier	Item_Visibility
# 118	DRA12	    0.041177505
# 1197	DRA12	    0.03493779266666667
# 1245	DRA12	    0.040911824
# 1693	DRA12	    0.03493779266666667
# 7467	DRA12	    0.041112694
# 8043	DRA12	    0.068535039
# 9023	DRA12	    0.040945898
# 12435	DRA12	    0.040747616
# 13604	DRA12	    0.041009558
# So for DRA12 the mean or avg is 0.034937792
# SO the Item_Visibility_MeanRatio for each row will be respective values/0.034937792
# e.g 0.041177505/0.034937792=0.9310779543761083
#A new feature is Created. As good features can drastically improve model performance and they invariably
#prove to be the difference between the best and the average model.

#2----Modifiying categories of 'Item_Fat_Content'
#In Item_Fat_Content there are only 2 types of Fat Given (LOW FAT, REGULAR FAT)
#But these are in Short-form or not in same form.
# Low Fat    8485
# Regular    4824
# LF          522
# reg         195
# low fat     178
#So combining all these content.
dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].replace({'LF':'Low Fat',
            'reg':'Regular',
            'low fat':'Low Fat'})

#There are some Non-Consumable product. So these product has a fat which is not
#good or Non_Edible. So these fat are counted in LOW & Regular Fat. So to seperate
#them we are creating a new category in 'Item_Fat_Content'.

dataset.loc[dataset['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
#Counting all the 3 types of Fat Content.
#print(dataset['Item_Fat_Content'].value_counts())

#3-----Determine the Years of operation of store
#The older the store the higher the sales. This tells which store is oldest
#as per 2013 data sales
dataset['Outlet_Years']=2013-dataset['Outlet_Establishment_Year']
#This will create a new column(2013-1985,2013-2009=28,4)

#4-----Numerical and One-Hot Coding of Categorical variables

#Since scikit-learn accepts only numerical variables, we have to convert
#all categories of nominal variables into numeric types. 
#Also, we want Outlet_Identifier as a variable as well. 
#So we creat a new variable ‘Outlet’ same as Outlet_Identifier and coded that.
#a=set(dataset['Outlet_Identifier'])#10 different 

#One-hot encoder only takes numerical categorical values, hence any value of 
#string type should be label encoded before one-hot encoded. 

#LabelEncoder
#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
dataset['Outlet'] = le.fit_transform(dataset['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])

#We get a row/1d array of range values e.g [0,0,1,2,1..]
#Now we need to create these in column
#0 1 2
#0 1 0
#1 0 0
#0 0 1...

#One-hot coding using pandas(preferred because it labels index column as well)   
dataset= pd.get_dummies(dataset, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])

#Alternative way for One-hot coding using sklearn(it does not label index column)
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#import numpy as np
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2,8,9,10,12,15])], remainder='passthrough')
#dataset2=np.array(ct.fit_transform(dataset))
#dataset2=pd.DataFrame(dataset2)
#dataset2.info()

#5.-----------Converting dataset into original form(Train/Test) 
#Final step is to convert data back into train and test data sets   
#dropping columns which are not used.
dataset.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

train = dataset.loc[dataset['source']=="Train_dataset"]
test = dataset.loc[dataset['source']=="Test_dataset"]

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions to the path or folder.
#train.to_csv("Train_modified_new_dataset.csv",index=False)
#test.to_csv("Test_modified_new_dataset.csv",index=False)


#----------5. MODEL BUILDING -------------

#1.------ Baseline Model-------
  #-> A baseline is the result of a very basic model/solution. One generally
  #   create a baseline and then try to make more complex solutions in order to
  #   get a better result. If the model gives a better score then it is a good
  #   model.
  #-> Baseline model is the one which requires no predictive model and its
  #   like an informed guess
  #-> Its like for comparing the new complex model 
#In this case lets predict the sales as the overall average sales

#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
#base1.to_csv("alg0.csv",index=False)

# 	Item_Identifier	Outlet_Identifier	Item_Outlet_Sales
# 0	   FDW58            OUT049	             2181.288913575032
# 1	   FDW14            OUT017	             2181.288913575032
# 2	   NCN55            OUT010	             2181.288913575032
# 3	   FDQ58            OUT017	             2181.288913575032
# .    .....            ......               .................

#2.-------- Creating a Function---

#Since there will be many models, instead of repeating the codes again 
#and again, I would like to define a generic function which takes the 
#algorithm and data as input and makes the model, performs train_test_split
#and generates submission.

#Define target and ID columns:
target = 'Item_Outlet_Sales'#(Y)
IDcol = ['Item_Identifier','Outlet_Identifier']

import sklearn.metrics as sm
global index
index=1
def modelfit(regressor, dtrain, dtest, predictors, target, IDcol, filename,name):
    #Fit the algorithm on the data
    regressor.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions =regressor.predict(dtrain[predictors])
    global index
    #Score of our model
    print('\n-----'+'{0}. '.format(index)+name.upper()+'-----')  

    index+=1
    
    print("Mean absolute error =", round(sm.mean_absolute_error(dtrain[target].values,dtrain_predictions ), 2)) 
    print("Mean squared error =", round(sm.mean_squared_error(dtrain[target].values, dtrain_predictions), 2)) 
    print("Median absolute error =", round(sm.median_absolute_error(dtrain[target].values, dtrain_predictions), 2)) 
    print("Explain variance score =", round(sm.explained_variance_score(dtrain[target].values, dtrain_predictions), 2)) 
    print("R2 score =", round(sm.r2_score(dtrain[target].values, dtrain_predictions), 2))
#-------------------------------------------------------------------------------------
#Mean absolute error: This is the average of absolute errors of all the data
                      #points in the given dataset.

#Mean squared error: This is the average of the squares of the errors of all 
                    #the data points in the given dataset. 
                    #It is one of the most popular metrics out there!

#Median absolute error: This is the median of all the errors in the given 
                        #dataset. The main advantage of this metric is that 
                        #it's robust to outliers. A single bad point in the 
                        #test dataset wouldn't skew the entire error metric, 
                        #as opposed to a mean error metric.

#Explained variance score: This score measures how well our model can 
                           #account for the variation in our dataset. 
                           #A score of 1.0 indicates that our model is 
                           #perfect.

#R2 score: This is pronounced as R-squared, and this score refers to the 
           #coefficient of determination. This tells us how well the 
           #unknown samples will be predicted by our model. 
           #The best possible score is 1.0, but the score 
           #can be negative as well.
#-------------------------------------------------------------------------------------
    #Predict on testing data:
    dtest[target] =regressor.predict(dtest[predictors])
    
    #Export submission file
    #This file contains the predicted values
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    
#3.---------- Applying Model to the Function---
import matplotlib.pyplot as plt

predictors = [x for x in train.columns if x not in [target]+IDcol]
length_predictors=len(predictors)
#List of independent variables(X1,X2.....)

#(1)--LINEAR REGRESSION--- 
from sklearn.linear_model import LinearRegression
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1_using_Linear_Regression.csv','Linear Regression')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
#Gives the coefficient of hpyothesis
coef1.plot(kind='bar', title='Model1 Coefficients',use_index=True)
plt.show()

#(2)-- DECISION TREE REGRESSION---
from sklearn.tree import DecisionTreeRegressor
alg2 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2_Decision_Tree_Regressor.csv','Decision Tree')
coef2 = pd.Series(alg2.feature_importances_, predictors).sort_values(ascending=False)
coef2.plot(kind='bar', title='Model2 Coefficients',use_index=True)
plt.show()

#(3)-- RANDOM FOREST REGRESSION---
from sklearn.ensemble import RandomForestRegressor
alg3 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3_Random_Forest_Regressor.csv','Random Forest')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Model3 Coefficients')
plt.show()

#(4)-- SUPPORT VECTOR MACHINE(RBF)---
from sklearn.svm import SVR
alg4 = SVR(kernel = 'rbf')
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4_SVM(SVR_Kernel-rbf).csv','Support Vector Regression(rbf)')
coef4=alg4._get_coef()
coef4=coef4.reshape(length_predictors)
coef4=pd.Series(coef4,predictors).sort_values()
coef4.plot(kind='bar',title='Model4 Coefficients',use_index=True)
plt.show()

#(5)-- SUPPORT VECTOR MACHINE(LINEAR)---
alg5 = SVR(kernel='linear')
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5_SVM(SVR_Kernel-linear).csv','Support Vector Regression(linear)')
coef5=alg5._get_coef()
coef5=coef5.reshape(length_predictors)
coef5=pd.Series(coef5,predictors).sort_values()
coef5.plot(kind='bar',title='Model5 Coefficients',use_index=True)
plt.show()

#(6)-- SUPPORT VECTOR MACHINE(POLY)---
alg6 = SVR(kernel = 'poly')
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6_SVM(SVR_Kernel-poly).csv','Support Vector Regression(poly)')
coef6=alg6._get_coef()
coef6=coef6.reshape(length_predictors)
coef6=pd.Series(coef6,predictors).sort_values()
coef6.plot(kind='bar',title='Model6 Coefficients',use_index=True)
plt.show()

#(7)-- RIDGE REGRESSION---
from sklearn.linear_model import Ridge
alg7 = Ridge(alpha=0.05,normalize=True)
modelfit(alg7, train, test, predictors, target, IDcol, 'alg7_Ridge_Regression.csv','Ridge Regression')
coef7 = pd.Series(alg7.coef_, predictors).sort_values()
coef7.plot(kind='bar', title='Model7 Coefficients',use_index=True)
plt.show()

#(8)-- XG-BOOST REGRESSION ---- (perfect model)
from xgboost import XGBRegressor
alg8=XGBRegressor(n_estimators=100,random_state=1234,max_depth=20)
modelfit(alg8, train, test, predictors, target, IDcol, 'alg8_XG-boost_Regression.csv','Xgboost Regression')
coef8 = pd.Series(alg8.feature_importances_, predictors).sort_values()
coef8.plot(kind='bar', title='Model8 Coefficients',use_index=True)
plt.show()
