import numpy as np


class SGD:

    def __init__(
        self,
        learning_rate=0.01
    ):

        self.learning_rate = learning_rate

    def update(
        self,
        weights,
        bias,
        dw,
        db
    ):

        weights -= (
            self.learning_rate * dw
        )

        bias -= (
            self.learning_rate * db
        )

        return (
            weights,
            bias
        )