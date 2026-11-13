"""Benchmarks for medoid initialization strategies and their scaling characteristics."""

import numpy as np
from clarans._initialization import (
    initialize_build,
    initialize_from_array,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)


class InitializationMethods:
    """Benchmark standalone medoid initialization algorithms directly."""

    params = ["k-medoids++", "random", "build", "heuristic", "explicit_array"]
    param_names = ["init_method"]

    def setup(self, init_method: str) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(500, 8)
        self.rng = np.random.RandomState(42)
        self.n_clusters = 5
        self.explicit_centers = self.X[:5].copy()

    def time_init(self, init_method: str) -> None:
        if init_method == "k-medoids++":
            initialize_k_medoids_plus_plus(self.X, self.n_clusters, self.rng, metric="euclidean")
        elif init_method == "random":
            self.rng.choice(len(self.X), self.n_clusters, replace=False)
        elif init_method == "build":
            initialize_build(self.X, self.n_clusters, metric="euclidean")
        elif init_method == "heuristic":
            initialize_heuristic(self.X, self.n_clusters, metric="euclidean")
        elif init_method == "explicit_array":
            initialize_from_array(
                self.explicit_centers, self.X, self.rng, n_clusters=self.n_clusters
            )


class InitializationScaling:
    """Benchmark initialization execution time scaling with sample size n."""

    params = ([200, 600], ["k-medoids++", "build", "heuristic"])
    param_names = ["n_samples", "init_method"]

    def setup(self, n_samples: int, init_method: str) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(n_samples, 8)
        self.rng = np.random.RandomState(42)
        self.n_clusters = 4

    def time_scaling(self, n_samples: int, init_method: str) -> None:
        if init_method == "k-medoids++":
            initialize_k_medoids_plus_plus(self.X, self.n_clusters, self.rng, metric="euclidean")
        elif init_method == "build":
            initialize_build(self.X, self.n_clusters, metric="euclidean")
        elif init_method == "heuristic":
            initialize_heuristic(self.X, self.n_clusters, metric="euclidean")
