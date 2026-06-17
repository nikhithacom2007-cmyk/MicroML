import numpy as np


class Adam:

    def __init__(
        self,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate

        self.beta1 = beta1
        self.beta2 = beta2

        self.epsilon = epsilon

        self.mw = None
        self.vw = None

        self.mb = 0
        self.vb = 0

        self.t = 0

    def update(
        self,
        weights,
        bias,
        dw,
        db
    ):

        if self.mw is None:

            self.mw = np.zeros_like(weights)
            self.vw = np.zeros_like(weights)

        self.t += 1

        self.mw = (
            self.beta1 * self.mw
            +
            (1 - self.beta1) * dw
        )

        self.mb = (
            self.beta1 * self.mb
            +
            (1 - self.beta1) * db
        )

        self.vw = (
            self.beta2 * self.vw
            +
            (1 - self.beta2) * (dw ** 2)
        )

        self.vb = (
            self.beta2 * self.vb
            +
            (1 - self.beta2) * (db ** 2)
        )

        mw_hat = (
            self.mw
            /
            (1 - self.beta1 ** self.t)
        )

        mb_hat = (
            self.mb
            /
            (1 - self.beta1 ** self.t)
        )

        vw_hat = (
            self.vw
            /
            (1 - self.beta2 ** self.t)
        )

        vb_hat = (
            self.vb
            /
            (1 - self.beta2 ** self.t)
        )

        weights -= (
            self.learning_rate
            *
            mw_hat
            /
            (np.sqrt(vw_hat) + self.epsilon)
        )

        bias -= (
            self.learning_rate
            *
            mb_hat
            /
            (np.sqrt(vb_hat) + self.epsilon)
        )

        return (
            weights,
            bias
        )