from sklearn.cluster import KMeans
import numpy as np

# Initialize the dataset
points = np.array([[0.1, 0.6],
                   [0.15, 0.71],
                   [0.08, 0.9],
                   [0.16, 0.85],
                   [0.2, 0.3],
                   [0.25, 0.5],
                   [0.24, 0.1],
                   [0.3, 0.2]])

C1= [0.1, 0.6]
C2= [0.3, 0.2]

kmean= KMeans(n_clusters= 2, init= np.array([C1, C2]))

kmean.fit(points)

p6_label= kmean.predict([[0.25,0.5]])
if p6_label == 0:
  print("p6 belongs to C1")
else:
  print("p6 belongs to C2")

labels= kmean.labels_
m2_population= np.sum(labels==1)
print(m2_population)

centroids= kmean.cluster_centers_
m1= centroids[0]
m2= centroids[1]
print(m1)
print(m2)