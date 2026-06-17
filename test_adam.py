import numpy as np

from microml.optimizers.adam import Adam


weights = np.array([5.0])

bias = 2.0

dw = np.array([1.5])

db = 0.5


optimizer = Adam(
    learning_rate=0.1
)


for step in range(5):

    weights, bias = optimizer.update(
        weights,
        bias,
        dw,
        db
    )

    print(
        f"Step {step+1}"
    )

    print(
        "Weight:",
        weights
    )

    print(
        "Bias:",
        bias
    )

    print("-" * 20)