"""
Tests and benchmarks for acceleration features:
- Incremental Medoid Distance Caching, Zero-Allocation Buffers.
- FastPAM1 Delta-Cost Updates with Pre-allocated Cython Buffers.
- Metric coverage, numerical stability, and bit-exact reproducibility.
"""

import time
import tracemalloc
import unittest
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from clarans import CLARANS, FastCLARANS, calculate_cost


class TestAccelerationFeatures(unittest.TestCase):
    """Test correctness, invariants, and reproducibility of acceleration features."""

    def setUp(self):
        self.X, _ = make_blobs(n_samples=300, n_features=4, centers=3, random_state=42)

    def test_clone_compatibility(self):
        """Cloning estimators must preserve parameters."""
        model = CLARANS(n_clusters=3, num_local=3)
        cloned = clone(model)
        self.assertEqual(cloned.n_clusters, 3)
        self.assertEqual(cloned.num_local, 3)

        fmodel = FastCLARANS(n_clusters=3, num_local=3)
        fcloned = clone(fmodel)
        self.assertEqual(fcloned.n_clusters, 3)
        self.assertEqual(fcloned.num_local, 3)

    def test_clarans_correctness_invariants(self):
        """CLARANS must produce valid clustering invariants."""
        model = CLARANS(n_clusters=3, num_local=4, max_neighbors=50, random_state=42)
        model.fit(self.X)

        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertEqual(len(np.unique(model.medoid_indices_)), 3)
        self.assertEqual(model.cluster_centers_.shape, (3, 4))
        self.assertEqual(len(model.labels_), len(self.X))
        expected_cost = calculate_cost(self.X, model.medoid_indices_, metric="euclidean")
        np.testing.assert_allclose(model.inertia_, expected_cost, rtol=1e-5)

    def test_fast_clarans_correctness_invariants(self):
        """FastCLARANS must produce valid clustering invariants."""
        model = FastCLARANS(n_clusters=3, num_local=4, max_neighbors=50, random_state=42)
        model.fit(self.X)

        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertEqual(len(np.unique(model.medoid_indices_)), 3)
        self.assertEqual(model.cluster_centers_.shape, (3, 4))
        self.assertEqual(len(model.labels_), len(self.X))
        expected_cost = calculate_cost(self.X, model.medoid_indices_, metric="euclidean")
        np.testing.assert_allclose(model.inertia_, expected_cost, rtol=1e-5)

    def test_incremental_caching_vs_brute_force_cost(self):
        """Incremental distance caching must yield exact same cost as recalculating."""
        for ModelClass in (CLARANS, FastCLARANS):
            m = ModelClass(n_clusters=4, num_local=2, max_neighbors=80, random_state=123)
            m.fit(self.X)

            recalculated_cost = calculate_cost(self.X, m.medoid_indices_, metric="euclidean")
            np.testing.assert_allclose(m.inertia_, recalculated_cost, rtol=1e-5)

    def test_deterministic_reproducibility(self):
        """Fixed random_state must produce identical medoids and inertia across runs."""
        for ModelClass in (CLARANS, FastCLARANS):
            m1 = ModelClass(
                n_clusters=3, num_local=4, max_neighbors=50, random_state=42
            ).fit(self.X)
            m2 = ModelClass(
                n_clusters=3, num_local=4, max_neighbors=50, random_state=42
            ).fit(self.X)

            np.testing.assert_array_equal(m1.medoid_indices_, m2.medoid_indices_)
            np.testing.assert_allclose(m1.inertia_, m2.inertia_, rtol=1e-12)

    def test_multi_metric_support(self):
        """FastCLARANS and CLARANS must work across all standard metrics."""
        for metric in ("cosine", "manhattan", "chebyshev", "euclidean"):
            m = FastCLARANS(
                n_clusters=3, num_local=2, max_neighbors=50, metric=metric, random_state=42
            ).fit(self.X)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertGreater(m.inertia_, 0.0)

    def test_precomputed_metric_support(self):
        """Precomputed distance matrix must run without error and match expectations."""
        D = pairwise_distances(self.X, metric="euclidean")
        for ModelClass in (CLARANS, FastCLARANS):
            model = ModelClass(
                n_clusters=3, num_local=2, max_neighbors=50, metric="precomputed", random_state=42
            )
            model.fit(D)
            self.assertEqual(len(model.medoid_indices_), 3)
            self.assertIsNone(model.cluster_centers_)
            self.assertGreater(model.inertia_, 0.0)


def run_benchmark():
    """Benchmark execution time and memory consumption for FastCLARANS vs CLARANS."""
    print("\n" + "=" * 70)
    print("ACCELERATION BENCHMARK: FastCLARANS vs CLARANS")
    print("=" * 70)

    for n_samples in [2000, 5000]:
        n_features = 8
        n_clusters = 5
        X, _ = make_blobs(
            n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42
        )

        print(f"\n--- Dataset: N = {n_samples:,}, Features = {n_features}, "
              f"Clusters = {n_clusters} ---")

        for ModelClass in [CLARANS, FastCLARANS]:
            tracemalloc.start()
            t0 = time.perf_counter()
            m = ModelClass(n_clusters=n_clusters, num_local=4, max_neighbors=100, random_state=42)
            m.fit(X)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"{ModelClass.__name__:<15} | Time: {elapsed*1000:6.1f} ms | "
                  f"Peak RAM: {peak/(1024**2):.2f} MB | Inertia: {m.inertia_:.2f}")


if __name__ == "__main__":
    unittest.main()
