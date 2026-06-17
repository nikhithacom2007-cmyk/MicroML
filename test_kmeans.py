import numpy as np

from microml.clustering.kmeans import (
    KMeans
)


X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [8, 8],
    [9, 9],
    [10, 10]
])


model = KMeans(
    k=2
)

model.fit(X)

labels = model.predict(X)

print(
    "Cluster Labels:"
)

print(
    labels
)

print(
    "\nCentroids:"
)

print(
    model.centroids
)