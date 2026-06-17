import numpy as np

from microml.optimizers.sgd import SGD


weights = np.array([5.0])

bias = 2.0

dw = np.array([1.5])

db = 0.5


optimizer = SGD(
    learning_rate=0.1
)

weights, bias = optimizer.update(
    weights,
    bias,
    dw,
    db
)

print("Updated Weight:")
print(weights)

print("Updated Bias:")
print(bias)