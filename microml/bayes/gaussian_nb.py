import numpy as np


class GaussianNB:

    def fit(self, X, y):

        self.classes = np.unique(y)

        n_features = X.shape[1]

        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:

            X_c = X[y == c]

            self.mean[c] = np.mean(
                X_c,
                axis=0
            )

            self.var[c] = np.var(
                X_c,
                axis=0
            )

            self.priors[c] = (
                len(X_c)
                /
                len(X)
            )

    def gaussian_pdf(
        self,
        class_idx,
        x
    ):

        mean = self.mean[class_idx]

        var = self.var[class_idx]

        numerator = np.exp(
            -(
                (x - mean) ** 2
            )
            /
            (
                2 * var
                + 1e-9
            )
        )

        denominator = np.sqrt(
            2
            * np.pi
            * var
            + 1e-9
        )

        return (
            numerator
            /
            denominator
        )

    def predict(self, X):

        predictions = []

        for x in X:

            posteriors = []

            for c in self.classes:

                prior = np.log(
                    self.priors[c]
                )

                likelihood = np.sum(
                    np.log(
                        self.gaussian_pdf(
                            c,
                            x
                        )
                        + 1e-9
                    )
                )

                posterior = (
                    prior
                    +
                    likelihood
                )

                posteriors.append(
                    posterior
                )

            predictions.append(
                self.classes[
                    np.argmax(
                        posteriors
                    )
                ]
            )

        return np.array(
            predictions
        )