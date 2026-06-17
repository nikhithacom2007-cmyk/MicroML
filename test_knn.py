import numpy as np

from microml.neighbors.knn import KNN

from microml.metrics.metrics import (
    accuracy
)


X_train = np.array([
    [1],
    [2],
    [3],
    [6],
    [7],
    [8]
])

y_train = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])


X_test = np.array([
    [2.5],
    [7.5]
])


model = KNN(k=3)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

print(
    "Predictions:"
)

print(
    predictions
)