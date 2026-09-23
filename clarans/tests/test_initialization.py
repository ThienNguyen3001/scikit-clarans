"""Unit and integration tests for medoid initialization algorithms.

Covers:
- k-medoids++ (probabilistic initialization, distance-weighted sampling)
- random (uniform random sampling)
- heuristic (distance-sum minimization)
- build (greedy PAM BUILD phase)
- explicit center arrays and array-like objects
- precomputed distance matrix initializations
- handling of duplicate centers and edge cases
"""

import unittest
import warnings

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from clarans import CLARANS
from clarans._initialization import (
    initialize_build,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)


class TestInitializationFunctions(unittest.TestCase):
    """Test individual initialization functions directly."""

    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    def test_random_init_via_model(self):
        """Random init should produce correct number of unique medoids."""
        for n_clusters in [2, 3, 5]:
            model = CLARANS(
                n_clusters=n_clusters, init="random", num_local=1,
                max_neighbors=10, random_state=42,
            )
            model.fit(self.X)
            self.assertEqual(len(model.medoid_indices_), n_clusters)
            self.assertEqual(len(np.unique(model.medoid_indices_)), n_clusters)
            self.assertTrue(all(0 <= idx < len(self.X) for idx in model.medoid_indices_))

    def test_heuristic_count_and_uniqueness(self):
        """Heuristic init should return correct number of unique medoids."""
        for n_clusters in [2, 3, 5]:
            medoids = initialize_heuristic(self.X, n_clusters, "euclidean")
            self.assertEqual(len(medoids), n_clusters)
            self.assertEqual(len(np.unique(medoids)), n_clusters)

    def test_heuristic_selects_smallest_sum_distance(self):
        """Heuristic should select points with smallest sum of distances."""
        medoids = initialize_heuristic(self.X, 3, "euclidean")
        D = pairwise_distances(self.X, metric="euclidean")
        dist_sums = np.sum(D, axis=1)
        expected = np.argsort(dist_sums)[:3]
        np.testing.assert_array_equal(medoids, expected)

    def test_build_count_and_uniqueness(self):
        """BUILD init should return correct number of unique medoids."""
        for n_clusters in [2, 3, 5]:
            medoids = initialize_build(self.X, n_clusters, "euclidean")
            self.assertEqual(len(medoids), n_clusters)
            self.assertEqual(len(np.unique(medoids)), n_clusters)

    def test_build_first_medoid_is_most_central(self):
        """BUILD should pick the most central point first."""
        medoids = initialize_build(self.X, 3, "euclidean")
        D = pairwise_distances(self.X, metric="euclidean")
        expected_first = np.argmin(D.sum(axis=1))
        self.assertEqual(medoids[0], expected_first)

    def test_build_sparse_precomputed_warning(self):
        """initialize_build should warn when a sparse precomputed matrix is passed."""
        from scipy.sparse import csr_matrix

        D = pairwise_distances(self.X[:10], metric="euclidean")
        D_sparse = csr_matrix(D)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            medoids = initialize_build(D_sparse, n_clusters=2, metric="precomputed")
            self.assertEqual(len(medoids), 2)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertIn("toarray", str(w[-1].message))
            self.assertIn("sparse", str(w[-1].message))

    def test_kmedoids_plusplus_count_and_uniqueness(self):
        """K-medoids++ should return correct number of unique medoids."""
        for n_clusters in [2, 3, 5]:
            rng = np.random.RandomState(42)
            medoids = initialize_k_medoids_plus_plus(
                self.X, n_clusters, rng, "euclidean"
            )
            self.assertEqual(len(medoids), n_clusters)
            self.assertEqual(len(np.unique(medoids)), n_clusters)

    def test_kmedoids_plusplus_indices_in_range(self):
        """K-medoids++ indices should be valid array indices."""
        rng = np.random.RandomState(42)
        medoids = initialize_k_medoids_plus_plus(self.X, 3, rng, "euclidean")
        self.assertTrue(all(0 <= idx < len(self.X) for idx in medoids))

    def test_kmedoids_plusplus_handles_zero_distance(self):
        """K-medoids++ should handle identical points gracefully."""
        X = np.array([
            [0, 0], [0, 0], [0, 0],
            [1, 1], [2, 2],
        ])
        rng = np.random.RandomState(42)
        medoids = initialize_k_medoids_plus_plus(X, 3, rng, "euclidean")
        self.assertEqual(len(medoids), 3)
        self.assertEqual(len(np.unique(medoids)), 3)


class TestInitializationDeep(unittest.TestCase):
    """Deep tests for initialization functions on precomputed, determinism, and duplicates."""

    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=60, centers=3, n_features=2, random_state=42
        )
        self.D = pairwise_distances(self.X, metric="euclidean")

    def test_heuristic_precomputed_matches_dense(self):
        """Heuristic init on precomputed D should match dense euclidean."""
        medoids_dense = initialize_heuristic(self.X, 3, metric="euclidean")
        medoids_precomp = initialize_heuristic(self.D, 3, metric="precomputed")
        np.testing.assert_array_equal(
            np.sort(medoids_dense), np.sort(medoids_precomp)
        )

    def test_build_precomputed_matches_dense(self):
        """BUILD init on precomputed D should match dense euclidean."""
        medoids_dense = initialize_build(self.X, 3, metric="euclidean")
        medoids_precomp = initialize_build(self.D, 3, metric="precomputed")
        np.testing.assert_array_equal(medoids_dense, medoids_precomp)

    def test_kmedoids_pp_precomputed_returns_valid_indices(self):
        """k-medoids++ on precomputed should return valid unique indices."""
        rng = np.random.RandomState(42)
        medoids = initialize_k_medoids_plus_plus(
            self.D, 3, rng, metric="precomputed"
        )
        self.assertEqual(len(medoids), 3)
        self.assertEqual(len(np.unique(medoids)), 3)
        self.assertTrue(np.all(medoids >= 0))
        self.assertTrue(np.all(medoids < len(self.D)))

    def test_kmedoids_pp_determinism(self):
        """Same seed should produce identical k-medoids++ results."""
        rng1 = np.random.RandomState(99)
        m1 = initialize_k_medoids_plus_plus(self.X, 3, rng1, "euclidean")
        rng2 = np.random.RandomState(99)
        m2 = initialize_k_medoids_plus_plus(self.X, 3, rng2, "euclidean")
        np.testing.assert_array_equal(m1, m2)

    def test_init_duplicate_centers_warns_and_fills(self):
        """When init centers map to duplicate medoid indices, warn and fill."""
        init_centers = np.array([
            self.X[0] + 1e-10,
            self.X[0] + 2e-10,
            self.X[10],
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = CLARANS(
                n_clusters=3, num_local=1, max_neighbors=10,
                init=init_centers, random_state=42,
            )
            model.fit(self.X)
            dup_warnings = [
                x for x in w
                if "duplicate" in str(x.message).lower()
            ]
            self.assertGreaterEqual(len(dup_warnings), 1)
            self.assertEqual(len(model.medoid_indices_), 3)
            self.assertEqual(len(np.unique(model.medoid_indices_)), 3)

    def test_heuristic_deterministic_multiple_calls(self):
        """Heuristic init is deterministic – two calls produce identical result."""
        m1 = initialize_heuristic(self.X, 3, "euclidean")
        m2 = initialize_heuristic(self.X, 3, "euclidean")
        np.testing.assert_array_equal(m1, m2)
