
## Starting with python

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

iris = datasets.load_iris()
print (iris.target_names)
print (type(iris))

wcss = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(iris.data[:, 1:4]) # Using only four columns with numerical data
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 15), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
C5_iris = kmeans.fit(iris.data[:, :4])
print(C5_iris)


iris.data = iris.data[:,:4]
print(iris.data)
print(C5_iris.labels_)
print(iris.data[C5_iris.labels_ == 0, 0])

#Visualising the clusters
plt.scatter(iris.data[C5_iris.labels_ == 0, 2], iris.data[C5_iris.labels_ == 0, 3], s = 100, c = 'red', label = 'Iris-C1')
plt.scatter(iris.data[C5_iris.labels_ == 1, 2], iris.data[C5_iris.labels_ == 1, 3], s = 100, c = 'green', label = 'Iris-C2')
plt.scatter(iris.data[C5_iris.labels_ == 2, 2], iris.data[C5_iris.labels_ == 2, 3], s = 100, c = 'yellow', label = 'Iris-C3')
plt.scatter(iris.data[C5_iris.labels_ == 3, 2], iris.data[C5_iris.labels_ == 3, 3], s = 100, c = 'blue', label = 'Iris-C4')
plt.scatter(iris.data[C5_iris.labels_ == 4, 2], iris.data[C5_iris.labels_ == 4, 3], s = 100, c = 'pink', label = 'Iris-C5')
#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s = 100, c = 'violet', label = 'Centroids')

plt.legend()

plt.show()



