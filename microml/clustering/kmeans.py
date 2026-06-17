import numpy as np


class KMeans:

    def __init__(
        self,
        k=2,
        max_iters=100
    ):

        self.k = k
        self.max_iters = max_iters

    def fit(self, X):

        n_samples = X.shape[0]

        random_indices = np.random.choice(
            n_samples,
            self.k,
            replace=False
        )

        self.centroids = X[
            random_indices
        ]

        for _ in range(
            self.max_iters
        ):

            clusters = [
                [] for _ in range(
                    self.k
                )
            ]

            for idx, sample in enumerate(X):

                distances = [
                    np.linalg.norm(
                        sample - centroid
                    )
                    for centroid in self.centroids
                ]

                cluster_idx = np.argmin(
                    distances
                )

                clusters[
                    cluster_idx
                ].append(idx)

            old_centroids = (
                self.centroids.copy()
            )

            for cluster_idx, cluster in enumerate(clusters):

                if len(cluster) > 0:

                    self.centroids[
                        cluster_idx
                    ] = np.mean(
                        X[cluster],
                        axis=0
                    )

            if np.allclose(
                old_centroids,
                self.centroids
            ):
                break

    def predict(self, X):

        labels = []

        for sample in X:

            distances = [
                np.linalg.norm(
                    sample - centroid
                )
                for centroid in self.centroids
            ]

            label = np.argmin(
                distances
            )

            labels.append(label)

        return np.array(
            labels
        )