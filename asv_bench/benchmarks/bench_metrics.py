"""Benchmarks for distance metric routing (dense, sparse, and precomputed)."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from clarans import FastCLARANS


class DistanceRouting:
    """Benchmark clustering across diverse supported data representations."""

    params = ["dense_euclidean", "dense_cosine", "sparse_csr", "precomputed"]
    param_names = ["metric_type"]

    def setup(self, metric_type: str) -> None:
        rng = np.random.RandomState(42)
        X_dense = rng.randn(400, 8)

        if metric_type == "dense_euclidean":
            self.data = X_dense
            self.metric = "euclidean"
        elif metric_type == "dense_cosine":
            self.data = X_dense
            self.metric = "cosine"
        elif metric_type == "sparse_csr":
            mask = rng.rand(*X_dense.shape) < 0.2
            self.data = csr_matrix(X_dense * mask)
            self.metric = "euclidean"
        elif metric_type == "precomputed":
            self.data = cdist(X_dense, X_dense, metric="euclidean")
            self.metric = "precomputed"

    def time_metric(self, metric_type: str) -> None:
        FastCLARANS(
            n_clusters=4,
            metric=self.metric,
            num_local=1,
            random_state=42,
        ).fit(self.data)
