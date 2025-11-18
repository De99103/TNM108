import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering


# Example dataset (10 points)
X2 = np.array([
    [5, 3],
    [10, 15],
    [15, 12],
    [24, 10],
    [30, 30],
    [85, 70],
    [71, 80],
    [60, 78],
    [70, 55],
    [80, 91]
])



customer_data = pd.read_csv('LAB1/shopping_data.csv')
print(customer_data.shape) # shape of the dataset
print(customer_data.head()) # print head of the dataset

data = customer_data.iloc[:, 3:5].values # Select specific columns
print(data[:5])     # Print first 5 rows
print (data.shape)


labels = range(1, 11)

plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(data[:,0],data[:,1], label='True Position')

for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom')
#plt.show()

linked = linkage(data, 'ward') # ward, single, complete, average. using ward  method 
labelList = range(1, 201)
plt.figure(figsize=(10, 7))
dendrogram(linked,
    orientation='top',
    labels=labelList,
    distance_sort='descending',
    show_leaf_counts=True)
plt.title("Dendrogram (Ward linkage)")
plt.show()

cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
plt.title("Agglomerative Clustering (Ward linkage)")
plt.show()