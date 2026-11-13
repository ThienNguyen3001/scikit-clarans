"""Benchmarks for CLARANS and FastCLARANS clustering."""

import numpy as np
from clarans import CLARANS, FastCLARANS


class TimeClustering:
    """Benchmark execution time of CLARANS vs FastCLARANS across sizes."""

    params = ([500, 1000], [3, 5])
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


class PeakMemClustering:
    """Benchmark peak memory usage of CLARANS vs FastCLARANS."""

    params = [500, 1000]
    param_names = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(n_samples, 10)

    def peakmem_fast_clarans(self, n_samples: int) -> None:
        FastCLARANS(
            n_clusters=4,
            num_local=2,
            random_state=42,
        ).fit(self.X)

    def peakmem_clarans(self, n_samples: int) -> None:
        CLARANS(
            n_clusters=4,
            num_local=2,
            random_state=42,
        ).fit(self.X)
