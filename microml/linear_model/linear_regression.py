import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None

        self.loss_history = []

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(
                X.T,
                (y_pred - y)
            )

            db = (1 / n_samples) * np.sum(
                y_pred - y
            )

            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )

            loss = np.mean(
                (y - y_pred) ** 2
            )

            self.loss_history.append(loss)

    def predict(self, X):

        return np.dot(
            X,
            self.weights
        ) + self.bias