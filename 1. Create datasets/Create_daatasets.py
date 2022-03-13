#Create dataset through visualisation.
#Dataset Generator


#1>----MAKE BLOBS
#->Generate blobs points with Gaussian distribution for clustering.
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from pandas import DataFrame

X_blobs,y_blobs=make_blobs(n_samples=100,centers=1,n_features=2,cluster_std=1)
df=DataFrame(dict(x=X_blobs[:,0],y=X_blobs[:,1],label=y_blobs))
plt.scatter(X_blobs[:,0], X_blobs[:,1], s=40,color='g')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()



#2>-----MAKE MOON


from sklearn.datasets import make_moons
X_moons,y_moons=make_moons(n_samples=1000,noise=0.01)
df=DataFrame(dict(x=X_moons[:,0],y=X_moons[:,1],label=y_moons))
plt.scatter(X_moons[:,0], X_moons[:,1], s=40,color='g')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()



#3>----MAKE CIRCLE

from sklearn.datasets import make_circles
X_circle,y_circle=make_circles(n_samples=1000,noise=0.01)
df=DataFrame(dict(x=X_circle[:,0],y=X_circle[:,1],label=y_circle))
plt.scatter(X_circle[:,0], X_circle[:,1], s=40,color='g')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()


#4>----LINEAR REGRESSION


from sklearn.datasets import make_regression
X,y=make_regression(n_samples=100,n_features=1,noise=0.1)
plt.scatter(X,y,color='g')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()
