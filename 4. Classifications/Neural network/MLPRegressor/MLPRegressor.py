import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

#Creating datasets
x=np.arange(0,5,0.01)# vector
x=x.reshape(-1,1)#Matrix

y=x**3
y=y.reshape(-1,1)
#y=y.reshape(2,3)-->gives a matrix of 2(rows)x4(column)
#y=y.reshape(-1,1)-->'-1' means the user don't know the dimension of the array and let numpy figure it out.
#(lenght of array)(rows)x1(column)(e.g nx1 matrix) 
plt.plot(x,y)




#Creating MLPRegressor
regressor=MLPRegressor(hidden_layer_sizes=(100),activation='logistic',solver='lbfgs',max_iter=10000)
#hidden_layers_sizes-->'100' means 1 hidden layer and 100 units(perceptrons)
                      #(7,1) means 1 hidden layer 7 units.
#solver(weight optimizations)--> for large dataset(>=1000) use 'adam' and for small datasets use 'lbfgs'.
#max_iter(solver iterates until convergence).
                      
#Training network
regressor.fit(x,y)

#predict network
y_pred=regressor.predict(x)

plt.plot(x,y_pred)
plt.legend(['original','predict'])
#The curves are very close to each other so its look like overlap.

#prediction error
error=np.sum(abs(y[:,0]-y_pred))/np.size(y)
print(error)
