import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('/content/drive/MyDrive/Mall_Customers.csv')

data.head()

data.shape

data.info()

data.isnull().sum()

x=data.iloc[:,[3,4]].values

from sklearn.cluster import KMeans

#Using the Elbow to firn out optimum no of custmers
wcss=[]
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++',random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
  
wcss

#Elbow curve plot
plt.plot(range(1,11),wcss)
plt.title("Enbow_method")
plt.xlabel("No_of_clusters")
plt.ylabel("WCSS")
plt.show()

#fir the k_means modl on the data
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)

#prediction
y_kmeans = kmeans.fit_predict(x)

#visualization the clusters
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans==0,1],s=100,c="red",label="cluster_1")
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans==1,1],s=100,c="blue",label="cluster_2")
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans==2,1],s=100,c="green",label="cluster_3")
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans==3,1],s=100,c="cyan",label="cluster_4")
plt.scatter(x[y_kmeans == 4,0], x[y_kmeans==4,1],s=100,c="magenta",label="cluster_5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300,c="yellow",label="Centroid")
