"""Benchmarks for core CLARANS and FastCLARANS clustering performance."""

import numpy as np
from clarans import CLARANS, FastCLARANS


class ClusteringRuntime:
    """Benchmark runtime scaling of FastCLARANS vs classic CLARANS."""

    params = ([500, 1500], [3, 8])
    param_names = ["n_samples", "n_clusters"]

    def setup(self, n_samples: int, n_clusters: int) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(n_samples, 10)

    def time_fast_clarans(self, n_samples: int, n_clusters: int) -> None:
        FastCLARANS(
            n_clusters=n_clusters,
            num_local=2,
            random_state=42,
        ).fit(self.X)

    def time_clarans(self, n_samples: int, n_clusters: int) -> None:
        CLARANS(
            n_clusters=n_clusters,
            num_local=2,
            random_state=42,
        ).fit(self.X)


class ClusteringMemory:
    """Benchmark memory footprint scaling with sample size to verify O(n) bounds."""

    params = [1000, 3000]
    param_names = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(n_samples, 10)

    def peakmem_fast_clarans(self, n_samples: int) -> None:
        FastCLARANS(
            n_clusters=5,
            num_local=2,
            random_state=42,
        ).fit(self.X)

    def peakmem_clarans(self, n_samples: int) -> None:
        CLARANS(
            n_clusters=5,
            num_local=2,
            random_state=42,
        ).fit(self.X)


class ParallelRestarts:
    """Benchmark multi-core scalability over Cython nogil blocks."""

    params = [1, 2]
    param_names = ["n_jobs"]

    def setup(self, n_jobs: int) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(1000, 10)

    def time_parallel_restarts(self, n_jobs: int) -> None:
        FastCLARANS(
            n_clusters=5,
            num_local=4,
            n_jobs=n_jobs,
            random_state=42,
        ).fit(self.X)
