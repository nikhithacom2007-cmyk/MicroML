import numpy as np

from microml.linear_model.linear_regression import LinearRegression

from microml.metrics.metrics import (
    rmse,
    r2_score
)


# Sample Dataset
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([
    3,
    5,
    7,
    9,
    11
])


# Create Model
model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)

# Train Model
model.fit(X, y)

# Predictions
predictions = model.predict(X)

print("Predictions:")
print(predictions)

print("\nRMSE:")
print(rmse(y, predictions))

print("\nR2 Score:")
print(r2_score(y, predictions))

print("\nWeight:")
print(model.weights)

print("\nBias:")
print(model.bias)



import matplotlib.pyplot as plt

# Plot actual data
plt.scatter(X, y, label="Actual Data")

# Plot prediction line
plt.plot(X, predictions, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression From Scratch")

plt.legend()

plt.show()


plt.figure(figsize=(8,5))

plt.plot(model.loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.show()


