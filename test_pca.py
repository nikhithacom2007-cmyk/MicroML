import numpy as np

from microml.decomposition.pca import PCA


X = np.array([
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
])


pca = PCA(
    n_components=1
)

X_reduced = (
    pca.fit_transform(X)
)

print(
    "Original Shape:",
    X.shape
)

print(
    "Reduced Shape:",
    X_reduced.shape
)

print(
    "\nTransformed Data:"
)

print(
    X_reduced
)