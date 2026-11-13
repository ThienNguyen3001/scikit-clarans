"""Benchmarks covering the full spectrum of distance metrics and data representations."""

from typing import Any
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csc_matrix, csr_matrix
from clarans import FastCLARANS


class DenseStandardMetrics:
    """Benchmark primary continuous dense metrics accelerated by SciPy cdist."""

    params = ["euclidean", "manhattan", "cosine", "chebyshev"]
    param_names = ["metric"]

    def setup(self, metric: str) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(300, 8)

    def time_dense_metric(self, metric: str) -> None:
        FastCLARANS(
            n_clusters=3,
            metric=metric,
            num_local=1,
            random_state=42,
        ).fit(self.X)


class DenseStatisticalMetrics:
    """Benchmark statistical and non-Euclidean dense metrics."""

    params = ["canberra", "braycurtis", "correlation", "minkowski_p3"]
    param_names = ["metric_name"]

    def setup(self, metric_name: str) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(300, 8)
        self.params: dict[str, Any] | None = None

        if metric_name == "minkowski_p3":
            self.metric = "minkowski"
            self.params = {"p": 3}
        else:
            self.metric = metric_name

    def time_statistical_metric(self, metric_name: str) -> None:
        FastCLARANS(
            n_clusters=3,
            metric=self.metric,
            metric_params=self.params,
            num_local=1,
            random_state=42,
        ).fit(self.X)


class SpecializedAndDomainMetrics:
    """Benchmark parametric, geospatial, missing-data, and custom callable metrics."""

    params = ["haversine", "mahalanobis", "nan_euclidean", "custom_callable"]
    param_names = ["metric_type"]

    def setup(self, metric_type: str) -> None:
        rng = np.random.RandomState(42)
        self.params: dict[str, Any] | None = None

        if metric_type == "haversine":
            # 2D spherical coordinates (latitude, longitude) in radians
            self.X = np.radians(rng.uniform(-60, 60, size=(300, 2)))
            self.metric = "haversine"
        elif metric_type == "mahalanobis":
            self.X = rng.randn(300, 6)
            cov = np.cov(self.X, rowvar=False) + np.eye(6) * 1e-4
            self.metric = "mahalanobis"
            self.params = {"VI": np.linalg.inv(cov)}
        elif metric_type == "nan_euclidean":
            self.X = rng.randn(300, 8)
            mask = rng.rand(*self.X.shape) < 0.05
            self.X[mask] = np.nan
            self.metric = "nan_euclidean"
        elif metric_type == "custom_callable":
            self.X = rng.randn(300, 8)

            def custom_manhattan(u: np.ndarray, v: np.ndarray) -> float:
                return float(np.sum(np.abs(u - v)))

            self.metric = custom_manhattan

    def time_specialized_metric(self, metric_type: str) -> None:
        FastCLARANS(
            n_clusters=3,
            metric=self.metric,
            metric_params=self.params,
            num_local=1,
            random_state=42,
        ).fit(self.X)


class SparseMatrixFormats:
    """Benchmark sparse matrix clustering comparing CSR vs CSC storage and sparsity ratios."""

    params = ["csr_10pct", "csr_1pct", "csc_10pct", "csc_1pct"]
    param_names = ["sparse_scenario"]

    def setup(self, sparse_scenario: str) -> None:
        rng = np.random.RandomState(42)
        dense = rng.randn(300, 15)

        density = 0.10 if "10pct" in sparse_scenario else 0.01
        mask = rng.rand(*dense.shape) < density
        sparse_data = dense * mask

        if "csr" in sparse_scenario:
            self.X = csr_matrix(sparse_data)
        else:
            self.X = csc_matrix(sparse_data)

    def time_sparse_format(self, sparse_scenario: str) -> None:
        FastCLARANS(
            n_clusters=3,
            metric="euclidean",
            num_local=1,
            random_state=42,
        ).fit(self.X)


class PrecomputedMatrices:
    """Benchmark precomputed pairwise distance matrices (symmetric vs asymmetric)."""

    params = ["symmetric", "asymmetric"]
    param_names = ["matrix_structure"]

    def setup(self, matrix_structure: str) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(300, 8)
        D = cdist(X, X, metric="euclidean")

        if matrix_structure == "symmetric":
            self.D = D
        elif matrix_structure == "asymmetric":
            D_asym = D.copy()
            D_asym[0, 1] += 10.0
            self.D = D_asym

    def time_precomputed_metric(self, matrix_structure: str) -> None:
        FastCLARANS(
            n_clusters=3,
            metric="precomputed",
            num_local=1,
            random_state=42,
        ).fit(self.D)
