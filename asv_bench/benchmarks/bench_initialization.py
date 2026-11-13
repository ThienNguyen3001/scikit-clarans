"""Benchmarks for medoid initialization strategies."""

import numpy as np
from clarans import FastCLARANS


class Initialization:
    """Benchmark medoid initialization algorithms."""

    params = ["k-medoids++", "random", "build", "heuristic"]
    param_names = ["init_method"]

    def setup(self, init_method: str) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(400, 8)

    def time_init(self, init_method: str) -> None:
        FastCLARANS(
            n_clusters=4,
            init=init_method,
            num_local=1,
            random_state=42,
        ).fit(self.X)
