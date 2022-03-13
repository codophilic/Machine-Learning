#This Project is based on Clustering Unsupervised Technique
"""
--------------------- DESCRIPTION ---------------------------

-->You are owing a supermarket mall and through membership cards ,
   you have some basic data about your customers like Customer ID, age, 
   gender, annual income and spending score.
   
-->Spending Score is something you assign to the customer based on your 
   defined parameters like customer behavior and purchasing data.

----------------$$ PROBLEM STATEMENT $$-----------------------

-->You own the mall and want to understand the customers like who can be 
   easily converge [Target Customers] so that the sense can be given to 
   marketing team and plan the strategy accordingly.

"""

#---------------- 1. LOADING THE DATASET ----------------------

import pandas as pd

Customers_dataset=pd.read_csv('Mall_Customers.csv') #(200,5)
#Customers_dataset['Annual Income (k$)'][0]=10
#--------------- 2. DATA ANALYSIS/EXPLORATION -----------------

import warnings
warnings.filterwarnings("ignore")

print("Statistical inference of the dataset\n")
describe_customer=Customers_dataset.describe()
#print(describe_customer)

#Checking if any missing values
missing_values=Customers_dataset.isnull().sum()

#---------------- 3. DATA VISUALISATION ------------------------

import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
"""
SEABORN:
    
-->seaborn is a visualization library for statistical graphics plotting.
-->It provides beautiful default styles and color palettes to make 
    statistical plots more attractive
-->It is built on the top of matplotlib library and also closely integrated 
    to the data structures of pandas 
   
"""

# 1.Count Plot of Gender

"""
-->'seaborn.countplot()' method is used to Show the counts of observation 
    in each categorical bin using bars

-->So in this case there are 2 categorical features (Male & Female). So 
    there are two bar plots 

-->figsize=(X,Y)-->'figsize' takes a tuple argument of two values
    X=Width, Y=Height of the figure in inches
   
-->'set_style' gives a background color to the plot with grid.
    style must be one of white, dark, whitegrid, darkgrid, ticks
   
-->'set_context' this affects things like the size of the labels, lines, 
    and other elements of the plot, but not the overall style
   
-->Remove the top and right spines from plot(s). This can be seen by putting
    set_style('ticks').
   
-->'annotate' gives the actual percentage or values on the top of
    the bar in the plot.
   
--> Every bar is know as a patch within a bar chart object. Therefore
    we create a loop iterating through every patches.

"""
plt.figure(figsize = (7,5))
sns.set_context('paper',font_scale=1.5)
sns.set(rc={'axes.facecolor':'greenyellow'})#background/theme color to plot
cp=sns.countplot(x = 'Gender' , data = Customers_dataset, palette=['#00FFFF','#FF00FF'])
sns.despine()
plt.title('1. COUNT PLOT OF GENDER')
plt.xlabel('Count')

for i in cp.patches:
    cp.annotate('    Total={:.1f}'.format(i.get_height()),(i.get_x(),i.get_height()))

plt.show()

#2. Distribution of Annual Income

"""
-->'distplot' method is the most common approach to visualizing a 
    distribution in the Histogram. A histogram is a bar plot where the axis 
    represent the data variable which is divided into sets of discrete bins
    and the count of observations falling within each bin is shown using height
    of the corresponding bar
    e.g if the bin size(binwidth)=3 so as per our dataset it will make a set
    of unique annual income and counts uts frequency/ observations
    15-2,16-2,17-2-->this will group into one bar which will have height as 
    6 units.
    if binwidth=1, for the above sample there will be 3 bar having 2 units
    heights for each bar.
   
-->'height'  & 'aspect' is like figsize

-->'KDE' stands for Kernel density Estimation.(kdeplot)
    ->A histograms aims to approximate the underlying probability density function
    that generated the data by binning and counting observations like above e.g.
    The same problem is solve by KDE, Rather than using discrete bins, a KDE 
    plots smooths observations with a Gaussian Kernel.
    ->The curvy line represents KDE.
    ->KDE represents the data using a continuous probability density curve  

"""
mean_or_avg_annual_income=Customers_dataset['Annual Income (k$)'].mean()
#sns.set_style('darkgrid')
sns.set(rc={'axes.facecolor':'sandybrown'})#background/theme color to plot
dp1=sns.displot(Customers_dataset['Annual Income (k$)'],color='#FF00FF',
                kde=True,binwidth=3,height=6,aspect=2)

#The below code helps to find out the mean on the curvy line
ax1=sns.kdeplot(Customers_dataset['Annual Income (k$)'],color='darkred')
kdeline1=ax1.lines[0]
xs1,ys1=kdeline1.get_xdata(),kdeline1.get_ydata()
h1=np.interp(mean_or_avg_annual_income,xs1,ys1)
ax1.vlines(mean_or_avg_annual_income,0,h1,color='darkred',linewidth=5)#vertical dotted lines
ax1.fill_between(xs1,0,ys1,facecolor='crimson',alpha=0.2)#underlying area of curve  

plt.title('2. Distribution of Annual Income', fontsize = 15)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count of observations of Annual Income')
plt.show()
"""
from the plot the solid lines shows the mean value of the KDE function
which is approx 60 and the calculated mean is 60.56 so this data can be 
use for setting the price value of the item, so that many people can afford 
it.(middle class family)
We also note that most of Mall Customers have Annual Income around 
50k-75k$.
"""

#3. Distribution of Age

mean_or_avg_Age=Customers_dataset['Age'].mean()
#sns.set_style('white')
sns.set(rc={'axes.facecolor':'black'})
dp2=sns.displot(Customers_dataset['Age'],color='#00FF00',
                kde=True,binwidth=3,height=6,aspect=2)

#The below code helps to find out the mean on the curvy line
ax2=sns.kdeplot(Customers_dataset['Age'],color='#3cb371')
kdeline2=ax2.lines[0]
xs2,ys2=kdeline2.get_xdata(),kdeline2.get_ydata()
h2=np.interp(mean_or_avg_Age,xs2,ys2)
ax2.vlines(mean_or_avg_Age,0,h2,color='#7cfc00',linewidth=5)#vertical dotted lines
ax2.fill_between(xs2,0,ys2,facecolor='#adff2f',alpha=0.2)#underlying area of curve  

plt.title('3. Distribution of Age', fontsize = 15)
plt.xlabel('Range of Age')
plt.ylabel('Count of observations of Age')
plt.show()

"""
The highest peak of age distribution is 38-40 which is mean distribution
so these age people often buys products from mall.
We notice that most of regular customers have age around 30-40 i.e middle age.
On the other hand elder and youngstres are not regular customers.
"""

#4. Distribution of Spending Score

mean_or_avg_Spending_score=Customers_dataset['Spending Score (1-100)'].mean()
#sns.set_style('white')
sns.set(rc={'axes.facecolor':'midnightblue'})
dp3=sns.displot(Customers_dataset['Spending Score (1-100)'],color='#7fffd4',
                kde=True,binwidth=3,height=6,aspect=2)

#The below code helps to find out the mean on the curvy line
ax3=sns.kdeplot(Customers_dataset['Spending Score (1-100)'],color='#87cefa')
kdeline3=ax3.lines[0]
xs3,ys3=kdeline3.get_xdata(),kdeline3.get_ydata()
h3=np.interp(mean_or_avg_Spending_score,xs3,ys3)
ax3.vlines(mean_or_avg_Spending_score,0,h3,color='#00ffff',linewidth=5)#vertical dotted lines
ax3.fill_between(xs3,0,ys3,facecolor='#87cefa',alpha=0.2)#underlying area of curve  

plt.title('4. Distribution of Spending Score (1-100)', fontsize = 15)
plt.xlabel('Range of Spending Score (1-100)')
plt.ylabel('Count of observations of Age')
plt.show()

#5. Finding Correlations between features by plotting HeatMap

"""
INTERPRETATION:
-->Each square shows the correlation between the variables on each axis. 
-->Correlation ranges from -1 to +1. Values closer to zero means there is no 
    linear trend between the two variables. 
-->The close to 1 the correlation is the more positively correlated they are 
    that is as one increases so does the other and the closer to 1 the stronger
    this relationship is. 
-->A correlation closer to -1 is similar, but instead of both increasing one 
    variable will decrease as the other increases. 
-->The diagonals are all 1/dark area because those squares are correlating 
    each variable to itself (so it's a perfect correlation). 
-->For the rest the larger the number and darker the color the higher the 
    correlation between the two variables. 
-->The plot is also symmetrical about the diagonal since the same two 
    variables are being paired together in those squares.
   
-->'Customers_dataset.corr()' is used to find the pairwise correlation of all
    columns in the dataframe. Any NA values in the dataframe are excluded 
    automatically. For any non-numeric datatype is ignored.
-->'corr()' method calculates correlations by using formula of covariance.
--> covariance is a relationship between two variables. The formula is
    cov(X1,X2)(sum(x1-mean(X1))*sum(x2-mean(X2)))*(1/(n-1))
    X1=column1,X2=column2,x1=column1 values,x2=column2 values,n=total no. of
    dataset.
-->This basic formula is used and some additional formula is added in Pearson's
    correlation, Spearman's correlations method,kendall correlations method.   

"""

corr = Customers_dataset.corr(method='pearson')
plt.figure(figsize = (8,6))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True,fmt='.2f',
            cmap = 'OrRd', linecolor='magenta',linewidth=3)
plt.title('Correlation of customer Features', y = 1.05, size=20)

#Here we get to know that customerID & Annual Income are highly correlated 
#which doesn't make any sense.
#Clearly we see the features are not well correlated with each other

#6. Pairplot of dataset. 

"""
-->A pairplot plot a pairwise relationships in a dataset. The pairplot function
    creates a grid of Axes such that each variable in data will be shared in
    the y-acis across a single row & in the x-axis across a single column.
--> N columns, NxN Matrices of plots

"""
sns.set(rc={'axes.facecolor':'honeydew'},style='ticks',color_codes=True)
sns.pairplot(Customers_dataset)
plt.title('6. Pairplot for the Customer Data', fontsize = 20)
plt.show()

#7. Scatter plot of Age v/s Annual Income

sns.set(rc={'axes.facecolor':'#4d148c'})
plt.figure(7,figsize=(9,8))
plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , 
            data = Customers_dataset,color='#ff6600') 
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('7. Age vs Annual Income',fontsize=15)

leg=plt.legend(fontsize=13,facecolor='lightyellow',framealpha=1)
leg.get_frame().set_edgecolor('lime')
leg.get_frame().set_linewidth(4)

plt.show()


#8. Scatter plot Age Vs Spending Score

sns.set(rc={'axes.facecolor':'lime'})
plt.figure(8,figsize=(9,8))
plt.scatter(x = 'Age' , y = 'Spending Score (1-100)' , 
            data = Customers_dataset,color='#293250') 
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('8. Age vs Spending Score (1-100)',fontsize=15)

leg=plt.legend(fontsize=13,facecolor='#fffafa',framealpha=1)
leg.get_frame().set_edgecolor('magenta')
leg.get_frame().set_linewidth(4)

plt.show()

"""
From above graph we can conclued that customers have age around 30-40 have 
more speding score than others. So they are most valueable customer 
of the Mall.
"""

#9. Scatter plot of Annual Income vs Spending Score

sns.set(rc={'axes.facecolor':'gold'})
plt.figure(9,figsize=(9,8))
plt.scatter(x = 'Annual Income (k$)' , y = 'Spending Score (1-100)' , 
            data = Customers_dataset,color='#9400d3') 
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('9. Annual Income v/s Spending Score (1-100)',fontsize=15)

leg=plt.legend(fontsize=13,facecolor='ivory',framealpha=1)
leg.get_frame().set_edgecolor('red')
leg.get_frame().set_linewidth(4)

plt.show()

#From above graph we can conclude that customers have 75k-100k have 
#more spending score than others.

#---------------4. MODEL BUILDING -----------------
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

global fignum
fignum=10

def elbow_method(X,clustername,plotname):
    wcss = []
    for i in range(1, 11):
        Kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        Kmeans.fit(X)
        wcss.append(Kmeans.inertia_)
        #The attribute 'Kmeans.inertia_' gives sum of squared distances of
        #samples to their closest cluster center.
        #So as No. of cluster increases the distances keeps reducing. 
        
    global fignum    
    plt.figure(fignum,figsize=(9,8))
    fignum+=1
    
    sns.set(rc={'axes.facecolor':'mintcream'})
    plt.plot(range(1, 11), wcss,linewidth=3,color='darkgreen')
    plt.title('The Elbow Method for '+clustername+' of plot '+plotname)
    plt.xlabel('Number of Cluster')
    plt.ylabel('WCSS')
    plt.show()
    
    return wcss
    

from scipy import interpolate
from scipy.spatial import ConvexHull
import numpy as np

"""
Here we are defining two different dataset copies so that one copied dataset
can be used for PART1(df1) and another copied dataset can be used for PART2(df)
(Both PARTS have DIFFERENT number of cluster )

"""

"""
Interpolation is a method for generating points between given points.
e.g suppose we have point 1 & point 2, we may interpolate and find some points
like 1.33 & 1.66. In ML Interpolate can be use to find the missing values.
It is often used to smooth the discrete points in a dataset.

The convex hull is the smallest set of connections between our data 
points to form a polygon that encloses all the points. We can even 
interpolate the lines of our polygon to make a smoother shape around our data.

HULL means the exterior or the shape of object, So a ConvexHull of a shape or
group of points is a tight fitting convex boundary around the points or the
shape

"""

def Convex_Hull_Boundry_Shape(df,cluster_KMeans,X1,X2,Y_pred,n):
    #Defining columns in df
    
    #Creating a 'cluster' name column
    df['cluster']=Y_pred
    #getting the centroids
    centroids=cluster_KMeans.cluster_centers_
    #Creating X & Y centroids list
    cen_x=[i[0] for i in centroids]
    cen_y=[i[1] for i in centroids]
    
    #Making a centroids X & Y columns in df1 where each classfied has its
    #respective cluster center.
    if n==5:
        df['cen_x']=df.cluster.map({
            0:cen_x[0],1:cen_x[1],2:cen_x[2],3:cen_x[3],4:cen_x[4]
            })
        
        df['cen_y']=df.cluster.map({
            0:cen_y[0],1:cen_y[1],2:cen_y[2],3:cen_y[3],4:cen_y[4]
            })
    else:
         df['cen_x']=df.cluster.map({
            0:cen_x[0],1:cen_x[1],2:cen_x[2],3:cen_x[3]
            })
         df['cen_y']=df.cluster.map({
            0:cen_y[0],1:cen_y[1],2:cen_y[2],3:cen_y[3]
            })
    #Each cluster is classified on the basis of colors and making this in 
    #seperate column.
    
    
    
    if n==5:  
        colors=['maroon','darkolivegreen','midnightblue','darkgoldenrod','purple']
        colors_bg=['red','green','blue','yellow','magenta']
        #boundary inside area background color
        color_centroids_stars=['brown','lime','cyan','darkorange','hotpink']
        df['c']=df.cluster.map({
            0:colors[0],1:colors[1],2:colors[2],3:colors[3],4:colors[4]
            })
        
    else:
        colors=['maroon','midnightblue','darkgoldenrod','purple']
        colors_bg=['red','blue','yellow','magenta']
        color_centroids_stars=['brown','cyan','darkorange','hotpink']
        df['c']=df.cluster.map({
            0:colors[0],1:colors[1],2:colors[2],3:colors[3]
            })
    
    global fignum
    sns.set(rc={'axes.facecolor':'azure'})
    plt.figure(fignum,figsize=(9,8))
    fignum+=1
        
    """
    alpha--> Blending value, between 0(transparent) & 1(opaque), applied for all
             shadow region and its respective datapoints.
    s-->centroid or datapoints size parameter (any value)        
    c-->color sequences for centroid / markers or datapoints, if 'c' is not 
        passed all the centroids or datapoints color will be same.
    """
    plt.scatter(df[X1],
            df[X2],
            c=df['c'],alpha=0.6,s=20)
    plt.scatter(cen_x,cen_y,marker='*',s=250,c=color_centroids_stars)
    
    for i in df.cluster.unique():
        """
        The points consists of 2D array of values of 'Annual Income (k$)' & 
        'Spending Score (1-100)' of respective clusters.
        (cluster 0 all X & Y, same for 
        1,2,3 & 4)
        """
        points=df[df.cluster==i][[X1,X2]].values
        
        hull=ConvexHull(points)
        #All x values and y values
        x_hull = np.append(points[hull.vertices,0],
                           points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                           points[hull.vertices,1][0])
        #This below line will give a polygon shape boundry line around the clusters
        #plt.fill(x_hull, y_hull, alpha=0.3, c=colors[i])
        
        #interpolate
        dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], 
                                        u=dist_along, s=1.5,k=3)
        #s->A smoothing condition.
        #K-> Order of the curve.(cubic,quadratic..etc)
        
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
        
        # plot shape
        plt.fill(interp_x, interp_y, '--', c=colors_bg[i], alpha=0.3)
        
    if n==5:
        interpretation=[
            'Standard Customers Group',
            'Careless Customers Group',
            'Target Customers Group',
            'Sensible Customers Group',
            'Carefull Customers Group'      
            ]
    else:
        interpretation=[
            'Customer have Potential',
            'Premium Customer Group',
            'Customer should be paid more attention',
            'Customer should be treated carefully'
            ]
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                       label='Cluster of {0}'.format(interpretation[i]), 
                       markerfacecolor=mcolor, markersize=10) 
                      for i, mcolor in enumerate(colors_bg)]
    
    legend_elements.extend([Line2D([0], [0], marker='*', color='w', 
                            label='Centroid - C{}'.format(i+1), 
                            markerfacecolor=mcolor, markersize=20)
                            for i, mcolor in enumerate(color_centroids_stars)])
    plt.legend(handles=legend_elements, loc='upper right', ncol=2)
    
    plt.xlim(0,140)
    plt.ylim(0,150)
    return cen_x,cen_y
    
    

from matplotlib.lines import Line2D

"""
Line2D connects all the vertices and marker at each vertex.
"""
    
def Lines_clustering(df,cluster,X1,X2,Y_pred,cen_x,cen_y,c,c1,c2,n):    
    
    global fignum
    plt.figure(fignum,figsize=(9,8))
    fignum+=1
    
    #defining new columns like previous
    #Here we use KMeans Centroids columns just to plot the lines 
    df['cluster']=Y_pred
    
    if n==5:
        colors_h=['red','green','blue','yellow','cyan']
        color_centroids_stars=['brown','lime','cyan','darkorange','hotpink']

    else:
        colors_h=['red','green','blue','yellow']
        color_centroids_stars=['brown','cyan','darkorange','hotpink']
    df['c']=c
    
    df['cen_x']=cen_x
    
    df['cen_y']=cen_y
    plt.scatter(df[X1],
                df[X2],
                c=df['c'],alpha=0.6,s=20)
 
    plt.scatter(c1,c2,marker='*',c=color_centroids_stars,s=200)
    
    #Here creating a new column with same values just because the X1 and 
    #and X2 strings have spaces in between which leads to error in 'val.' line.
    #So taking first word.
    
    df['firstvariable']=df[X1]
    df['secondvariable']=df[X2]
    
    for i, val in df.iterrows():
       
        x1 = [val.firstvariable, val.cen_x]
        x2 = [val.secondvariable, val.cen_y]
        
        plt.plot(x1, x2, c=val.c, alpha=0.2,linewidth=2,linestyle='-')
        #linestyle-->'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 
        #            'dashdot', 'dotted'
    
    if n==5:
        interpretation=[
            'Standard Customers Group',
            'Careless Customers Group',
            'Target Customers Group',
            'Sensible Customers Group',
            'Carefull Customers Group'      
            ]
    else:
        interpretation=[
            'Customer have Potential',
            'Premium Customer Group',
            'Customer should be paid more attention',
            'Customer should be treated carefully'
            ]
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                       label='Cluster of {0}'.format(interpretation[i]), 
                       markerfacecolor=mcolor, markersize=10) 
                      for i, mcolor in enumerate(colors_h)]
    
    #marker-->dataset point/centroid shape 
    #color-->label notation color of symbol
    #markerfacecolor-->label color
    #makerssize-->label symbol size
    
    legend_elements.extend([Line2D([0], [0], marker='*', color='w', 
                            label='Centroid - C{}'.format(i+1), 
                            markerfacecolor=mcolor, markersize=20)
                            for i, mcolor in enumerate(color_centroids_stars)])
    
    plt.legend(handles=legend_elements, loc='upper right', ncol=2)
    
    # x and y limits
    plt.xlim(0,170)
    plt.ylim(0,170)

    
#--(1. Annual income and Spending score )
#We first apply cluster solution between Annual income and Spending score

#---(A) Applying K-Means Clustering Method
X = Customers_dataset.iloc[:, [3, 4]].values

#Finding number of cluster using elbow method
wcss_1A_Kmeans=elbow_method(X,"K-Means","Annual income and Spending score")

#Applying K-means 
cluster1_KMeans=KMeans(n_clusters = 5, init = 'k-means++', 
               max_iter = 300, n_init =10, random_state = 0)
#Optimal cluster is 5
Y1_pred_KMeans=cluster1_KMeans.fit_predict(X)

#Plotting Visualisation using ConvexHull

#Makes a shallow copy
df1=Customers_dataset.copy() 

c1_1,c2_1=Convex_Hull_Boundry_Shape(df1,cluster1_KMeans,
                        'Annual Income (k$)','Spending Score (1-100)',Y1_pred_KMeans,5)
#c1,c2-->Getting the centroids to use in Hierarchical Clustering

#---(B) Applying Hierarchical Clustering Method


#Finding number of cluster using dendrogram
# import scipy.cluster.hierarchy as sch

# dendrogram1= sch.dendrogram(sch.linkage(X, method = 'ward'))
# plt.title('Dendogram1')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean Distance')
# plt.show()

#Applying Agglomerative Hierarchical Clustering
cluster1_Agg=AgglomerativeClustering(n_clusters = 5, 
                                  affinity = 'euclidean', linkage = 'ward')
Y1_pred_Agg=cluster1_Agg.fit_predict(X)

#Plotting Visualisation using Lines
df2=Customers_dataset.copy()
 
Lines_clustering(df2,cluster1_Agg,'Annual Income (k$)',
                 'Spending Score (1-100)',Y1_pred_Agg,
                 df1['cen_x'],df1['cen_y'],df1['c'],c1_1,c2_1,5)

"""
From both graph we can see our previous prediction is also right. 
Mall has customers whos income around 75k-100k they are most valueable
customer.
"""





#--(2. Age and Spending Score)
#Now we apply cluster solution between Age and Spending Score

#---(A) Applying K-Means Clustering Method
X = Customers_dataset.iloc[:, [2, 4]].values

#Finding number of cluster using elbow method
wcss_2A_Kmeans=elbow_method(X,"K-Means","Age and Spending Score")

#Applying K-means 
cluster2_KMeans=KMeans(n_clusters = 4, init = 'k-means++', 
               max_iter = 300, n_init =10, random_state = 0)
#Optimal cluster is 4
Y2_pred_KMeans=cluster2_KMeans.fit_predict(X)

#Plotting Visualisation
df3=Customers_dataset.copy()

c1_2,c2_2=Convex_Hull_Boundry_Shape(df3,cluster2_KMeans,
                          'Age','Spending Score (1-100)',Y2_pred_KMeans,4)


#---(B) Applying Hierarchical Clustering Method


#Finding number of cluster using dendrogram
# import scipy.cluster.hierarchy as sch

# dendrogram2= sch.dendrogram(sch.linkage(X, method = 'ward'))
# plt.figure(fignum,figsize=(10,8))
# fignum+=1
# plt.title('Dendogram2')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean Distance')
# plt.show()

#Applying Agglomerative Hierarchical Clustering 
cluster2_Agg=AgglomerativeClustering(n_clusters = 4, 
                                  affinity = 'euclidean', linkage = 'ward')
Y2_pred_Agg=cluster2_Agg.fit_predict(X)

#Plotting Visualisation

df4=Customers_dataset.copy()

Lines_clustering(df4,cluster2_Agg,'Age',
                 'Spending Score (1-100)',Y2_pred_Agg,
                 df3['cen_x'],df3['cen_y'],df3['c'],c1_2,c2_2,4)

"""
From above graph we note middle age customers are most valuable for Mall, 
which is also previously preidcted.
"""










