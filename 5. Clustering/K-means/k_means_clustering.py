# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

#There is a big mall in a specific city thst contains the information of its client
#that subscribe to the membership card. When the client subscribe to the card 
#they are provided information like customere ID,Age,Annual income & gender.
#Since the have this card they use it to buy all sorts of things in the mall.
#And therefore the mall has the puirchase history of each of its client member
#and therefore the have the spending column as record and scaled from 1-100.
#The closer the spending score to 1 the less the client spends and the closer the 
#value to 100 the more the client spends.
#Now we have to segments into two different groups i.e Annual income and spending score.
#Here we dont have any idea in how many different segments these two category can be 
#more divided so typically a clustering problems.

X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = [] #Within cluster sum of squares

#Trying K values from 1-11 and then finding the optimal value of K. 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #init-->initialization method =k-means++.
    #max_iter=300(default)-->Maximum no. of iterations there can be to find the final cluster.
    #n_init=10(default)--->No. of times K-means algorithm runs with different initialize of centroids..
    #random_state--->It fixes all the random factors of the K-means process.
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #----> computes within cluster sum of squares
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X) #First it will fit X for 5 clusters and then 
#it will return the values in which  all the datapoints of X belongs to one of these
#5 clusters.
#So customer 1 belongs to cluster No. 4, customer 2 belongs to cluster No. 3 and so on.

# Visualising the clusters

#5 cluster -->0,1..4

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
#In X 2D array-->e.g x_coordinate(0)=15,y-coordinate(1)=39 which is of cluster for 4 y_kmeans==4.
#X[Cluster No. from y_kmeans=[0..4],x_coordinate],X[Cluster No. from y_kmeans=[0..4],y_coordinate]
#s--->size for datapoints or its like diameter size of the points
#c-->color for the specific cluster
#Repeating for other 4 clusters.
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
#cluster_centers gives the list of all centroid of the clusters.
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#Ctrl+L or %clear-->clear console
#%reset -f ---> remove all variables
