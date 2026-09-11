import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    pairwise_distances,
    pairwise_distances_argmin_min,
    silhouette_score,
)

from clarans import CLARANS
from clarans.initialization import (
    initialize_build,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)
from clarans.utils import calculate_cost


class TestCLARANS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(
            n_samples=100, centers=3, n_features=2, random_state=42
        )

    def test_fit(self):
        clarans = CLARANS(n_clusters=3, num_local=2, max_neighbors=10, random_state=42)
        clarans.fit(self.X)

        self.assertEqual(len(clarans.cluster_centers_), 3)
        self.assertEqual(len(clarans.labels_), 100)
        self.assertTrue(hasattr(clarans, "medoid_indices_"))

        for idx in clarans.medoid_indices_:
            self.assertTrue(np.any(np.all(self.X == self.X[idx], axis=1)))

    def test_predict(self):
        clarans = CLARANS(n_clusters=3, num_local=1, random_state=42)
        clarans.fit(self.X)
        labels = clarans.predict(self.X)
        self.assertEqual(labels.shape, (100,))

    def test_convergence(self):
        clarans = CLARANS(n_clusters=3, num_local=10, max_neighbors=100, random_state=42)
        clarans.fit(self.X)
        score = silhouette_score(self.X, clarans.labels_)
        self.assertGreater(score, -0.1, f"Silhouette score too low: {score}")

    def test_init_methods(self):
        """Test different initialization methods."""
        for init_method in ["random", "heuristic", "k-medoids++", "build"]:
            clarans = CLARANS(
                n_clusters=3,
                num_local=1,
                max_neighbors=10,
                init=init_method,
                random_state=42,
            )
            clarans.fit(self.X)
            self.assertEqual(len(clarans.cluster_centers_), 3)
            self.assertEqual(len(clarans.labels_), 100)

    def test_precomputed_init(self):
        """Test initialization with precomputed array."""
        init_centers = self.X[[0, 10, 20]]
        clarans = CLARANS(
            n_clusters=3, num_local=1, max_neighbors=10, init=init_centers, random_state=42
        )
        clarans.fit(self.X)
        self.assertEqual(len(clarans.cluster_centers_), 3)

    def test_metrics(self):
        """Test different metrics."""
        for metric in ["euclidean", "manhattan"]:
            clarans = CLARANS(
                n_clusters=3, num_local=1, max_neighbors=10, metric=metric, random_state=42
            )
            clarans.fit(self.X)
            self.assertEqual(len(clarans.cluster_centers_), 3)

    def test_input_validation_init(self):
        """Test invalid init parameter."""
        clarans = CLARANS(n_clusters=3, init="invalid_method")
        with self.assertRaises(ValueError):
            clarans.fit(self.X)

        clarans = CLARANS(n_clusters=3, init=self.X[:2])
        with self.assertRaises(ValueError):
            clarans.fit(self.X)

    def test_transform(self):
        """Test transform() returns distances to cluster centers."""
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=10, random_state=42)
        clarans.fit(self.X)
        X_transformed = clarans.transform(self.X)

        # Shape: (n_samples, n_clusters)
        self.assertEqual(X_transformed.shape, (100, 3))
        # All distances should be non-negative
        self.assertTrue(np.all(X_transformed >= 0))
        # Each medoid should have distance 0 to itself
        for i, idx in enumerate(clarans.medoid_indices_):
            self.assertAlmostEqual(X_transformed[idx, i], 0.0, places=10)

    def test_cosine_metric(self):
        """Test that cosine metric works correctly."""
        clarans = CLARANS(
            n_clusters=3, num_local=1, max_neighbors=10, metric="cosine", random_state=42
        )
        clarans.fit(self.X)
        self.assertEqual(len(clarans.cluster_centers_), 3)
        self.assertEqual(len(clarans.labels_), 100)
        labels = clarans.predict(self.X)
        np.testing.assert_array_equal(labels, clarans.labels_)

    def test_single_cluster(self):
        """Test with n_clusters=1 (edge case)."""
        clarans = CLARANS(n_clusters=1, num_local=1, max_neighbors=10, random_state=42)
        clarans.fit(self.X)
        self.assertEqual(len(clarans.cluster_centers_), 1)
        self.assertTrue(np.all(clarans.labels_ == 0))

    def test_inertia_attribute(self):
        """Test that inertia_ is set after fit and is non-negative."""
        clarans = CLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=42)
        clarans.fit(self.X)
        self.assertTrue(hasattr(clarans, "inertia_"))
        self.assertGreaterEqual(clarans.inertia_, 0)

    def test_n_iter_and_n_swaps_attributes(self):
        """Test that n_iter_ and n_swaps_ are set correctly after fit."""
        clarans = CLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=42)
        clarans.fit(self.X)
        self.assertTrue(hasattr(clarans, "n_iter_"))
        self.assertTrue(hasattr(clarans, "n_swaps_"))
        self.assertGreaterEqual(clarans.n_iter_, 1)
        self.assertGreaterEqual(clarans.n_swaps_, 0)
        self.assertLessEqual(clarans.n_swaps_, clarans.n_iter_)

    def test_medoid_indices_sorted(self):
        """Test that medoid_indices_ is always sorted."""
        clarans = CLARANS(n_clusters=4, num_local=5, max_neighbors=30, random_state=42)
        clarans.fit(self.X)
        self.assertTrue(np.all(np.diff(clarans.medoid_indices_) >= 0))

    def test_sparse_input(self):
        """Test CLARANS with scipy sparse matrices and arrays."""
        try:
            from scipy import sparse
        except Exception:
            self.skipTest("scipy not available")

        # Test with csr_matrix
        X_sparse = sparse.csr_matrix(self.X)
        model = CLARANS(n_clusters=3, num_local=1, random_state=42)
        model.fit(X_sparse)
        labels = model.predict(X_sparse)
        self.assertEqual(labels.shape, (100,))

        # Test with csr_array if available
        if hasattr(sparse, "csr_array"):
            X_arr = sparse.csr_array(self.X)
            model_arr = CLARANS(n_clusters=3, num_local=1, random_state=42)
            model_arr.fit(X_arr)
            labels_arr = model_arr.predict(X_arr)
            self.assertEqual(labels_arr.shape, (100,))

    def test_medoid_uniqueness_multiple_seeds(self):
        """All medoids should be unique across multiple random seeds."""
        for seed in range(10):
            clarans = CLARANS(
                n_clusters=3, num_local=1, max_neighbors=30, random_state=seed
            )
            clarans.fit(self.X)
            unique_count = len(np.unique(clarans.medoid_indices_))
            self.assertEqual(
                unique_count, 3,
                f"Seed {seed}: Expected 3 unique medoids, got {unique_count}",
            )

    def test_cluster_centers_match_medoids(self):
        """cluster_centers_ should be the actual data points at medoid_indices_."""
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=30, random_state=42)
        clarans.fit(self.X)
        for i, center in enumerate(clarans.cluster_centers_):
            medoid_idx = clarans.medoid_indices_[i]
            np.testing.assert_array_equal(
                center, self.X[medoid_idx],
                f"Cluster center {i} doesn't match X[{medoid_idx}]",
            )

    def test_labels_in_valid_range(self):
        """Labels should be in range [0, n_clusters)."""
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=30, random_state=42)
        clarans.fit(self.X)
        self.assertTrue(
            np.all((clarans.labels_ >= 0) & (clarans.labels_ < 3)),
            "All labels should be in [0, n_clusters)",
        )

    def test_predict_matches_fit_labels(self):
        """predict(X) on training data should match labels_."""
        clarans = CLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=42)
        clarans.fit(self.X)
        predicted = clarans.predict(self.X)
        np.testing.assert_array_equal(
            predicted, clarans.labels_,
            "predict(X) should match labels_ on training data",
        )

    def test_determinism(self):
        """Same random_state should give identical results."""
        clarans1 = CLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=123)
        clarans1.fit(self.X)

        clarans2 = CLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=123)
        clarans2.fit(self.X)

        np.testing.assert_array_equal(
            clarans1.medoid_indices_, clarans2.medoid_indices_,
            "Same random_state should give same medoids",
        )
        np.testing.assert_array_equal(
            clarans1.labels_, clarans2.labels_,
            "Same random_state should give same labels",
        )

    def test_max_neighbors_default(self):
        """Default max_neighbors should be 'auto' and calculated correctly."""
        clarans = CLARANS(n_clusters=3, num_local=1, random_state=42)
        self.assertEqual(clarans.max_neighbors, "auto")
        clarans.fit(self.X)
        expected = max(250, int(0.0125 * 3 * (100 - 3)))
        self.assertEqual(clarans.max_neighbors_, expected)

    def test_max_neighbors_explicit_auto(self):
        """Explicit max_neighbors='auto' should work identically to default."""
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors="auto", random_state=42)
        clarans.fit(self.X)
        expected = max(250, int(0.0125 * 3 * (100 - 3)))
        self.assertEqual(clarans.max_neighbors_, expected)

    def test_max_neighbors_custom(self):
        """Custom max_neighbors should be used when provided."""
        clarans = CLARANS(n_clusters=3, max_neighbors=100, random_state=42)
        clarans.fit(self.X)
        self.assertEqual(clarans.max_neighbors_, 100)

    def test_n_clusters_exceeds_samples(self):
        """Should raise error when n_clusters >= n_samples."""
        clarans = CLARANS(n_clusters=150)
        with self.assertRaises(ValueError):
            clarans.fit(self.X)

    def test_predict_before_fit(self):
        """Should raise error when predicting before fit."""
        clarans = CLARANS(n_clusters=3)
        with self.assertRaises(Exception):
            clarans.predict(self.X)

    def test_keyword_only_args(self):
        """CLARANS should enforce keyword-only arguments per SLEP009."""
        with self.assertRaises(TypeError):
            CLARANS(3)

    def test_get_feature_names_out(self):
        """CLARANS should provide get_feature_names_out per SLEP007."""
        clarans = CLARANS(n_clusters=3)
        with self.assertRaises(NotFittedError):
            clarans.get_feature_names_out()

        clarans.fit(self.X)
        names = clarans.get_feature_names_out()
        np.testing.assert_array_equal(
            names, np.array(["clarans0", "clarans1", "clarans2"], dtype=object)
        )

    def test_pandas_output(self):
        """CLARANS should support set_output(transform='pandas') per SLEP018."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")

        clarans = CLARANS(n_clusters=3, random_state=42)
        clarans.set_output(transform="pandas")
        clarans.fit(self.X)
        transformed = clarans.transform(self.X)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertListEqual(
            list(transformed.columns), ["clarans0", "clarans1", "clarans2"]
        )


class TestCLARANSEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_identical_points(self):
        """Algorithm should handle dataset with all identical points."""
        X = np.ones((50, 2))
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=20, random_state=42)
        clarans.fit(X)
        unique_medoids = len(np.unique(clarans.medoid_indices_))
        self.assertEqual(unique_medoids, 3)

    def test_two_clusters(self):
        """Algorithm should work with 2 clusters."""
        X, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=42)
        clarans = CLARANS(n_clusters=2, num_local=1, max_neighbors=20, random_state=42)
        clarans.fit(X)
        self.assertEqual(len(clarans.medoid_indices_), 2)
        self.assertEqual(len(np.unique(clarans.labels_)), 2)

    def test_high_dimensional(self):
        """Algorithm should work with high-dimensional data."""
        X, _ = make_blobs(n_samples=100, centers=3, n_features=50, random_state=42)
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=30, random_state=42)
        clarans.fit(X)
        self.assertEqual(len(clarans.medoid_indices_), 3)
        self.assertEqual(clarans.cluster_centers_.shape, (3, 50))

    def test_n_clusters_close_to_n_samples(self):
        """Algorithm should work when n_clusters is close to n_samples."""
        X = np.random.RandomState(42).randn(20, 2)
        clarans = CLARANS(n_clusters=15, num_local=1, max_neighbors=10, random_state=42)
        clarans.fit(X)
        self.assertEqual(len(clarans.medoid_indices_), 15)
        self.assertEqual(len(np.unique(clarans.medoid_indices_)), 15)

    def test_single_feature(self):
        """Algorithm should work with single feature."""
        X = np.random.RandomState(42).randn(50, 1)
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=20, random_state=42)
        clarans.fit(X)
        self.assertEqual(len(clarans.medoid_indices_), 3)
        self.assertEqual(clarans.cluster_centers_.shape, (3, 1))


class TestInitializationFunctions(unittest.TestCase):
    """Test initialization functions directly."""

    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

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


class TestCostCalculation(unittest.TestCase):
    """Test the calculate_cost utility function."""

    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    def test_cost_is_non_negative(self):
        """Cost should always be non-negative."""
        medoids = np.array([0, 30, 60])
        cost = calculate_cost(self.X, medoids, "euclidean")
        self.assertGreaterEqual(cost, 0)

    def test_cost_matches_manual_calculation(self):
        """Cost should equal sum of minimum distances to medoids."""
        medoids = np.array([0, 30, 60])
        cost = calculate_cost(self.X, medoids, "euclidean")
        medoid_points = self.X[medoids]
        _, min_dists = pairwise_distances_argmin_min(
            self.X, medoid_points, metric="euclidean"
        )
        expected = np.sum(min_dists)
        self.assertAlmostEqual(cost, expected, places=10)

    def test_cost_zero_when_all_points_are_medoids(self):
        """Cost should be zero when every point is a medoid."""
        X_small = np.array([[0, 0], [1, 1], [2, 2]])
        medoids = np.array([0, 1, 2])
        cost = calculate_cost(X_small, medoids, "euclidean")
        self.assertEqual(cost, 0)

    def test_cost_with_different_metrics(self):
        """Cost calculation should work with different metrics."""
        medoids = np.array([0, 30, 60])
        for metric in ["euclidean", "manhattan"]:
            cost = calculate_cost(self.X, medoids, metric)
            _, min_dists = pairwise_distances_argmin_min(
                self.X, self.X[medoids], metric=metric
            )
            expected = np.sum(min_dists)
            self.assertAlmostEqual(cost, expected, places=10)

    def test_cost_precomputed(self):
        """calculate_cost should work with metric='precomputed'."""
        D = pairwise_distances(self.X, metric="euclidean")
        medoids = np.array([0, 30, 60])
        cost = calculate_cost(D, medoids, metric="precomputed")
        expected = calculate_cost(self.X, medoids, metric="euclidean")
        self.assertAlmostEqual(cost, expected, places=5)


class TestCLARANSValidationAndPrecomputed(unittest.TestCase):
    """Tests for parameter validation and precomputed metric support."""

    def setUp(self):
        self.X, _ = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)
        self.D = pairwise_distances(self.X, metric="euclidean")

    def test_invalid_n_clusters(self):
        """n_clusters < 1 should raise ValueError."""
        for val in [0, -1, -5]:
            with self.assertRaises(ValueError):
                CLARANS(n_clusters=val).fit(self.X)

    def test_invalid_num_local(self):
        """num_local < 1 should raise ValueError."""
        for val in [0, -1]:
            with self.assertRaises(ValueError):
                CLARANS(num_local=val).fit(self.X)

    def test_invalid_max_neighbors(self):
        """Invalid max_neighbors (<= 0, None, float, bool, or bad str) should raise ValueError."""
        for val in [0, -1, None, "invalid", 1.5, True, False]:
            with self.assertRaises(ValueError):
                CLARANS(max_neighbors=val).fit(self.X)

    def test_removed_max_iter(self):
        """max_iter was removed from the API; passing it should raise TypeError."""
        with self.assertRaises(TypeError):
            CLARANS(max_iter=10)

    def test_precomputed_metric_fit_predict_transform(self):
        """CLARANS should support metric='precomputed'."""
        model = CLARANS(
            n_clusters=3,
            num_local=2,
            max_neighbors=20,
            metric="precomputed",
            random_state=42,
        )
        model.fit(self.D)

        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertEqual(len(model.labels_), 50)
        self.assertGreaterEqual(model.inertia_, 0)
        self.assertIsNone(model.cluster_centers_)
        self.assertFalse(hasattr(model, "n_features_in_"))

        # predict on full square matrix
        labels = model.predict(self.D)
        np.testing.assert_array_equal(labels, model.labels_)

        # transform on full square matrix
        X_trans = model.transform(self.D)
        self.assertEqual(X_trans.shape, (50, 3))
        self.assertTrue(np.all(X_trans >= 0))

        # transform on submatrix of medoids
        D_sub = self.D[:, model.medoid_indices_]
        labels_sub = model.predict(D_sub)
        np.testing.assert_array_equal(labels_sub, model.labels_)

    def test_precomputed_non_square(self):
        """Precomputed metric should require a square matrix."""
        non_square = self.X  # 50 x 2
        model = CLARANS(n_clusters=3, metric="precomputed", random_state=42)
        with self.assertRaises(ValueError):
            model.fit(non_square)

    def test_precomputed_all_init_methods(self):
        """All init methods should work with precomputed metric."""
        for init_method in ["random", "heuristic", "k-medoids++", "build"]:
            model = CLARANS(
                n_clusters=3,
                num_local=1,
                max_neighbors=10,
                init=init_method,
                metric="precomputed",
                random_state=42,
            )
            model.fit(self.D)
            self.assertEqual(len(model.medoid_indices_), 3)

    def test_deterministic_init_warning_and_caching(self):
        """CLARANS should warn when num_local > 1 with deterministic init and succeed."""
        import warnings
        for init_strategy in ["heuristic", "build", self.X[:3]]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = CLARANS(
                    n_clusters=3, num_local=2, max_neighbors=10, init=init_strategy, random_state=42
                )
                model.fit(self.X)
                self.assertTrue(any(issubclass(warn.category, UserWarning) for warn in w))
                self.assertEqual(len(model.medoid_indices_), 3)

    def test_cache_parameter(self):
        """Test cache parameter: True, False, and invalid values."""
        model_cached = CLARANS(
            n_clusters=3, num_local=1, max_neighbors=20, cache=True, random_state=42
        )
        model_cached.fit(self.X)
        self.assertEqual(len(model_cached.medoid_indices_), 3)

        model_non_cached = CLARANS(
            n_clusters=3, num_local=1, max_neighbors=20, cache=False, random_state=42
        )
        model_non_cached.fit(self.X)
        self.assertEqual(len(model_non_cached.medoid_indices_), 3)

        np.testing.assert_array_equal(
            model_cached.medoid_indices_, model_non_cached.medoid_indices_
        )
        self.assertAlmostEqual(model_cached.inertia_, model_non_cached.inertia_, places=5)

        with self.assertRaises(ValueError):
            CLARANS(cache="invalid").fit(self.X)

    def test_legacy_parameters_removed(self):
        """Passing removed numlocal or maxneighbor should raise TypeError."""
        with self.assertRaises(TypeError):
            CLARANS(numlocal=2)
        with self.assertRaises(TypeError):
            CLARANS(maxneighbor=25)

    def test_parameters_num_local_max_neighbors(self):
        """Using num_local and max_neighbors should work and set max_neighbors_."""
        model = CLARANS(
            n_clusters=3, num_local=2, max_neighbors=25, random_state=42
        )
        model.fit(self.X)
        self.assertEqual(model.max_neighbors_, 25)
        self.assertFalse(hasattr(model, "maxneighbor_"))


if __name__ == "__main__":
    unittest.main()
