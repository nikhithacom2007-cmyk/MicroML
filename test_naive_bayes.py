import numpy as np

from microml.bayes.gaussian_nb import (
    GaussianNB
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


model = GaussianNB()

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