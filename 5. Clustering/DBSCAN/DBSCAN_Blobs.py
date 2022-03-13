# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 22:15:53 2021

@author: Admin
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=0)
df=pd.DataFrame(dict(x=X[:,0],y=X[:,1],label=y))
plt.scatter(X[:,0], X[:,1], s=5,color='g')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()


#In DBSCAN we need to choose right value of epsilon(eps).
#We find a suitable value of 'eps' by calculating the distance to the nearest 
#'n' points for each points,sorting and plotting the results.
#Then look where the change is most varied(Elbow method like)

from sklearn.neighbors import NearestNeighbors
neigh=NearestNeighbors(n_neighbors=2)
nbrs=neigh.fit(X)
distance,indices=nbrs.kneighbors(X)
distance=np.sort(distance,axis=0)
distance=distance[:,1]
#plt.plot(distance)

#Fit the training set for DBSCAN

from sklearn.cluster import DBSCAN 
cluster=DBSCAN(eps=0.3,min_samples=3)
cluster.fit(X)
No_of_cluster=cluster.labels_
colors=['blue','red','green']
vectorize=np.vectorize(lambda z:colors[z%len(colors)])
plt.scatter(X[:,0],X[:,1],c=vectorize(No_of_cluster))