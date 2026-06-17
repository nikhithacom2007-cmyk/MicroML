import numpy as np

from microml.linear_model.logistic_regression import (
    LogisticRegression
)

from microml.metrics.metrics import (
    accuracy
)


X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])


model = LogisticRegression(
    learning_rate=0.1,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)

print("Predictions:")
print(predictions)

print("\nAccuracy:")
print(
    accuracy(
        y,
        predictions
    )
)

print("\nWeights:")
print(model.weights)

print("\nBias:")
print(model.bias)