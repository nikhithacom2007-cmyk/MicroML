import numpy as np

from microml.optimizers.momentum import Momentum


weights = np.array([5.0])

bias = 2.0

dw = np.array([1.5])

db = 0.5


optimizer = Momentum(
    learning_rate=0.1,
    beta=0.9
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