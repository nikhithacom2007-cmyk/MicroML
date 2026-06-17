import numpy as np


class Momentum:

    def __init__(
        self,
        learning_rate=0.01,
        beta=0.9
    ):

        self.learning_rate = learning_rate
        self.beta = beta

        self.vw = None
        self.vb = 0

    def update(
        self,
        weights,
        bias,
        dw,
        db
    ):

        if self.vw is None:

            self.vw = np.zeros_like(
                weights
            )

        self.vw = (
            self.beta * self.vw
            +
            (1 - self.beta) * dw
        )

        self.vb = (
            self.beta * self.vb
            +
            (1 - self.beta) * db
        )

        weights -= (
            self.learning_rate * self.vw
        )

        bias -= (
            self.learning_rate * self.vb
        )

        return (
            weights,
            bias
        )