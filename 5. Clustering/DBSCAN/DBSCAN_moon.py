#DBSCAN

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
from sklearn.datasets import make_moons
X,y=make_moons(n_samples=400,noise=0.05)
df=pd.DataFrame(dict(x=X[:,0],y=X[:,1],label=y))
plt.scatter(X[:,0], X[:,1], s=5,color='g')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()


#In DBSCAN we need to choose right value of epsilon(eps).
#We find a suitable value of 'eps' by calculating the distance to the nearest 
#'n' points for each points,sorting and plotting the results.
#Then look where the change is most varied(Elbow method like)

#Choose the point where the rate of increase in value jumps drastically and
#that ppoint is called elbow point.
#Here we took above the point of curve elbow point
from sklearn.neighbors import NearestNeighbors
neigh=NearestNeighbors(n_neighbors=2)
nbrs=neigh.fit(X)
distance,indices=nbrs.kneighbors(X)
distance=np.sort(distance,axis=0)
distance=distance[:,1]
#plt.plot(distance)




#Fit the training set for DBSCAN

from sklearn.cluster import DBSCAN 
cluster=DBSCAN(eps=0.125,min_samples=3)
cluster.fit(X)
No_of_cluster=cluster.labels_
colors=['blue','red','black']
vectorize=np.vectorize(lambda z:colors[z%len(colors)])
plt.scatter(X[:,0],X[:,1],c=vectorize(No_of_cluster))
plt.legend()
plt.show()
















