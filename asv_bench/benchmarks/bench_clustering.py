"""Benchmarks covering core clustering algorithms, scaling, datatypes, and parallelism."""

import numpy as np
from clarans import CLARANS, FastCLARANS


class AlgorithmComparison:
    """Benchmark FastCLARANS vs CLARANS (delta tracking) vs CLARANS (brute force)."""

    params = ["FastCLARANS", "CLARANS_delta", "CLARANS_brute_force"]
    param_names = ["algorithm"]

    def setup(self, algorithm: str) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(600, 10)

    def time_algorithm(self, algorithm: str) -> None:
        if algorithm == "FastCLARANS":
            FastCLARANS(n_clusters=4, num_local=2, random_state=42).fit(self.X)
        elif algorithm == "CLARANS_delta":
            CLARANS(
                n_clusters=4,
                num_local=2,
                cost_evaluation="delta",
                random_state=42,
            ).fit(self.X)
        elif algorithm == "CLARANS_brute_force":
            CLARANS(
                n_clusters=4,
                num_local=2,
                cost_evaluation="brute_force",
                random_state=42,
            ).fit(self.X)


class ClusterScaling:
    """Benchmark scalability as cluster count k grows from small to large."""

    params = ([600], [3, 8, 20])
    param_names = ["n_samples", "n_clusters"]

    def setup(self, n_samples: int, n_clusters: int) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(n_samples, 10)

    def time_fast_clarans(self, n_samples: int, n_clusters: int) -> None:
        FastCLARANS(
            n_clusters=n_clusters,
            num_local=1,
            random_state=42,
        ).fit(self.X)

    def time_clarans(self, n_samples: int, n_clusters: int) -> None:
        CLARANS(
            n_clusters=n_clusters,
            num_local=1,
            random_state=42,
        ).fit(self.X)


class DimensionScaling:
    """Benchmark clustering performance from low to high feature dimensionality."""

    params = [2, 20, 100]
    param_names = ["n_features"]

    def setup(self, n_features: int) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(500, n_features)

    def time_dimension_scaling(self, n_features: int) -> None:
        FastCLARANS(n_clusters=4, num_local=1, random_state=42).fit(self.X)


class ParallelScalability:
    """Benchmark multi-threaded speedup across local search restarts over Cython nogil."""

    params = [1, 2, -1]
    param_names = ["n_jobs"]

    def setup(self, n_jobs: int) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(800, 10)

    def time_parallel_restarts(self, n_jobs: int) -> None:
        FastCLARANS(
            n_clusters=5,
            num_local=4,
            n_jobs=n_jobs,
            random_state=42,
        ).fit(self.X)


class DataTypesAndLayouts:
    """Benchmark fused numeric precision (float64 vs float32) and memory layout."""

    params = ["float64_C", "float32_C", "float64_Fortran"]
    param_names = ["dtype_layout"]

    def setup(self, dtype_layout: str) -> None:
        rng = np.random.RandomState(42)
        X_raw = rng.randn(600, 10)

        if dtype_layout == "float64_C":
            self.X = np.ascontiguousarray(X_raw, dtype=np.float64)
        elif dtype_layout == "float32_C":
            self.X = np.ascontiguousarray(X_raw, dtype=np.float32)
        elif dtype_layout == "float64_Fortran":
            self.X = np.asfortranarray(X_raw, dtype=np.float64)

    def time_dtype_layout(self, dtype_layout: str) -> None:
        FastCLARANS(n_clusters=4, num_local=2, random_state=42).fit(self.X)


class ClusteringMemory:
    """Benchmark memory footprint scaling with sample size to verify bounded O(n) RAM."""

    params = [1000, 3000, 5000]
    param_names = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        rng = np.random.RandomState(42)
        self.X = rng.randn(n_samples, 10)

    def peakmem_fast_clarans(self, n_samples: int) -> None:
        FastCLARANS(n_clusters=5, num_local=1, random_state=42).fit(self.X)

    def peakmem_clarans(self, n_samples: int) -> None:
        CLARANS(n_clusters=5, num_local=1, random_state=42).fit(self.X)
