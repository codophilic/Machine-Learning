
import pandas as pd
from sklearn.datasets import load_boston

boston=load_boston()
dataset=pd.DataFrame(
    boston.data,
    columns=boston.feature_names
    )
print(dataset.info())
X=dataset.iloc[:,:-1]
Y=boston.target

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

regression1=Ridge()
alphas={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,55,60,66,70,100]}
gridsearch_Ridge=GridSearchCV(estimator=regression1,
    param_grid=alphas,
    scoring='neg_mean_squared_error',
    cv=5
    )

gridsearch_Ridge.fit(X,Y)

print("FOR RIDGE REGRESSION\n")
print(gridsearch_Ridge.best_params_)
print(gridsearch_Ridge.best_score_)
print()
"""

The 'neg_mean_squared_error' value must be closer to 0.
More closer to 0 more the model is good.

"""

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

regression2=Lasso()
alphas={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,0.4,0.25,1,5,10,50,55,100]}
gridsearch_Lasso=GridSearchCV(estimator=regression2,
    param_grid=alphas,
    scoring='neg_mean_squared_error',
    cv=5
    )

gridsearch_Lasso.fit(X,Y)

print("FOR LASSO REGRESSION\n")
print(gridsearch_Lasso.best_params_)
print(gridsearch_Lasso.best_score_)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=0)

predict_Ridge=gridsearch_Ridge.predict(X_test)
predict_Lasso=gridsearch_Lasso.predict(X_test)



import matplotlib.pyplot as plt

plt.plot(sorted(Y_test),color='red',label='Test Output',linewidth=3)
plt.plot(sorted(predict_Ridge),'-',color='green',label='Ridge Ouput',linewidth=2)
plt.plot(sorted(predict_Lasso),'-',color='yellow',label='Lasso Ouput',linewidth=2)
plt.legend()
plt.show()










