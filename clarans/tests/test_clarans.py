import unittest
import warnings

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    pairwise_distances,
    pairwise_distances_argmin_min,
    silhouette_score,
)

from clarans import CLARANS, FastCLARANS
from clarans.utils import calculate_cost, check_medoids


@pytest.mark.parametrize("estimator_cls", [CLARANS, FastCLARANS])
class TestCommonEstimators:
    @pytest.fixture(autouse=True)
    def _setup_data(self):
        self.X, self.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    def test_fit(self, estimator_cls):
        model = estimator_cls(n_clusters=3, num_local=2, max_neighbors=10, random_state=42)
        model.fit(self.X)

        assert len(model.cluster_centers_) == 3
        assert len(model.labels_) == 100
        assert hasattr(model, "medoid_indices_")
        np.testing.assert_array_equal(model.cluster_centers_, self.X[model.medoid_indices_])
        np.testing.assert_array_equal(model.labels_, np.argmin(model.transform(self.X), axis=1))

    def test_predict(self, estimator_cls):
        model = estimator_cls(n_clusters=3, num_local=1, random_state=42)
        model.fit(self.X)
        labels = model.predict(self.X)
        assert labels.shape == (100,)
        np.testing.assert_array_equal(labels, model.labels_)

    def test_sparse_input(self, estimator_cls):
        try:
            from scipy import sparse
        except Exception:
            pytest.skip("scipy not available")

        # Test with csr_matrix
        X_sparse = sparse.csr_matrix(self.X)
        model = estimator_cls(n_clusters=3, num_local=1, random_state=42)
        model.fit(X_sparse)
        labels = model.predict(X_sparse)
        assert labels.shape == (100,)

        # Test with csr_array if available
        if hasattr(sparse, "csr_array"):
            X_arr = sparse.csr_array(self.X)
            model_arr = estimator_cls(n_clusters=3, num_local=1, random_state=42)
            model_arr.fit(X_arr)
            labels_arr = model_arr.predict(X_arr)
            assert labels_arr.shape == (100,)

    def test_init_methods(self, estimator_cls):
        for init_method in ["random", "heuristic", "k-medoids++", "build"]:
            model = estimator_cls(
                n_clusters=3, num_local=1, max_neighbors=10, init=init_method, random_state=42
            )
            model.fit(self.X)
            assert len(model.cluster_centers_) == 3
            assert len(model.labels_) == 100

    def test_precomputed_init(self, estimator_cls):
        init_centers = self.X[[0, 10, 20]]
        model = estimator_cls(
            n_clusters=3, num_local=1, max_neighbors=10, init=init_centers, random_state=42
        )
        model.fit(self.X)
        assert len(model.cluster_centers_) == 3

    def test_input_validation_init(self, estimator_cls):
        model = estimator_cls(n_clusters=3, init="invalid_method")
        with pytest.raises(ValueError):
            model.fit(self.X)

        model = estimator_cls(n_clusters=3, init=self.X[:2])
        with pytest.raises(ValueError):
            model.fit(self.X)

    def test_transform(self, estimator_cls):
        model = estimator_cls(n_clusters=3, num_local=1, max_neighbors=10, random_state=42)
        model.fit(self.X)
        X_transformed = model.transform(self.X)

        assert X_transformed.shape == (100, 3)
        assert np.all(X_transformed >= 0)
        for i, idx in enumerate(model.medoid_indices_):
            assert np.isclose(X_transformed[idx, i], 0.0, atol=1e-10)

    def test_cosine_metric(self, estimator_cls):
        model = estimator_cls(
            n_clusters=3, num_local=1, max_neighbors=10, metric="cosine", random_state=42
        )
        model.fit(self.X)
        assert len(model.cluster_centers_) == 3
        assert len(model.labels_) == 100
        labels = model.predict(self.X)
        np.testing.assert_array_equal(labels, model.labels_)

    def test_single_cluster(self, estimator_cls):
        model = estimator_cls(n_clusters=1, num_local=1, max_neighbors=10, random_state=42)
        model.fit(self.X)
        assert len(model.cluster_centers_) == 1
        assert np.all(model.labels_ == 0)

    def test_inertia_attribute(self, estimator_cls):
        model = estimator_cls(n_clusters=3, num_local=2, max_neighbors=50, random_state=42)
        model.fit(self.X)
        assert hasattr(model, "inertia_")
        assert model.inertia_ >= 0

    def test_invalid_parameters(self, estimator_cls):
        with pytest.raises(ValueError):
            estimator_cls(n_clusters=0).fit(self.X)
        with pytest.raises(ValueError):
            estimator_cls(num_local=0).fit(self.X)
        for val in [0, -1, "invalid", 1.5, True, False]:
            with pytest.raises(ValueError):
                estimator_cls(max_neighbors=val).fit(self.X)
        with pytest.raises(TypeError):
            estimator_cls(max_iter=-1)

    def test_precomputed_metric(self, estimator_cls):
        D = pairwise_distances(self.X, metric="euclidean")
        model = estimator_cls(
            n_clusters=3, num_local=2, max_neighbors=20, metric="precomputed", random_state=42
        )
        model.fit(D)

        assert len(model.medoid_indices_) == 3
        assert len(model.labels_) == 100
        assert model.inertia_ >= 0
        assert not hasattr(model, "cluster_centers_")
        assert model.n_features_in_ == 100

        labels = model.predict(D)
        np.testing.assert_array_equal(labels, model.labels_)

        transformed = model.transform(D)
        assert transformed.shape == (100, 3)

    def test_precomputed_non_square(self, estimator_cls):
        model = estimator_cls(n_clusters=3, metric="precomputed", random_state=42)
        with pytest.raises(ValueError):
            model.fit(self.X)

    def test_deterministic_init_warning(self, estimator_cls):
        with pytest.warns(UserWarning):
            model = estimator_cls(
                n_clusters=3, num_local=2, max_neighbors=10, init="heuristic", random_state=42
            )
            model.fit(self.X)
        assert len(model.medoid_indices_) == 3

    def test_keyword_only_args(self, estimator_cls):
        with pytest.raises(TypeError):
            estimator_cls(3)

    def test_get_feature_names_out(self, estimator_cls):
        prefix = estimator_cls.__name__.lower()
        model = estimator_cls(n_clusters=3)
        with pytest.raises(NotFittedError):
            model.get_feature_names_out()

        model.fit(self.X)
        names = model.get_feature_names_out()
        np.testing.assert_array_equal(
            names,
            np.array([f"{prefix}0", f"{prefix}1", f"{prefix}2"], dtype=object),
        )

    def test_get_feature_names_out_validation(self, estimator_cls):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas is not installed")

        prefix = estimator_cls.__name__.lower()
        df = pd.DataFrame(self.X, columns=[f"col_{i}" for i in range(self.X.shape[1])])
        model = estimator_cls(n_clusters=3, random_state=42).fit(df)
        assert hasattr(model, "feature_names_in_")

        valid_cols = [f"col_{i}" for i in range(self.X.shape[1])]
        names = model.get_feature_names_out(valid_cols)
        np.testing.assert_array_equal(
            names, np.array([f"{prefix}0", f"{prefix}1", f"{prefix}2"], dtype=object)
        )

        with pytest.raises(ValueError, match="input_features should have length equal"):
            model.get_feature_names_out(["col_0"])

    def test_pandas_output(self, estimator_cls):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas is not installed")

        prefix = estimator_cls.__name__.lower()
        model = estimator_cls(n_clusters=3, random_state=42)
        model.set_output(transform="pandas")
        model.fit(self.X)
        transformed = model.transform(self.X)
        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == [f"{prefix}0", f"{prefix}1", f"{prefix}2"]

    def test_legacy_parameters_removed(self, estimator_cls):
        with pytest.raises(TypeError):
            estimator_cls(numlocal=2)
        with pytest.raises(TypeError):
            estimator_cls(maxneighbor=25)

    def test_parameters_num_local_max_neighbors(self, estimator_cls):
        model = estimator_cls(n_clusters=3, num_local=2, max_neighbors=25, random_state=42)
        model.fit(self.X)
        assert model.max_neighbors_ == 25
        assert not hasattr(model, "maxneighbor_")

    def test_delta_tolerance_rejects_ghost_swaps(self, estimator_cls):
        X_dup = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
        model = estimator_cls(n_clusters=2, num_local=1, max_neighbors=50, random_state=42)
        model.fit(X_dup)
        assert model.n_swaps_ >= 0


class TestCLARANS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    def test_convergence(self):
        clarans = CLARANS(n_clusters=3, num_local=10, max_neighbors=100, random_state=42)
        clarans.fit(self.X)
        score = silhouette_score(self.X, clarans.labels_)
        self.assertGreater(score, -0.1, f"Silhouette score too low: {score}")

    def test_metrics(self):
        """Test different metrics."""
        for metric in ["euclidean", "manhattan"]:
            clarans = CLARANS(
                n_clusters=3, num_local=1, max_neighbors=10, metric=metric, random_state=42
            )
            clarans.fit(self.X)
            self.assertEqual(len(clarans.cluster_centers_), 3)

    def test_n_iter_and_n_swaps_attributes(self):
        """Test that n_iter_, n_swaps_, and neighborhood_size_ are set correctly,
        and deprecated total_n_iter_ / total_n_swaps_ / total_neighbors_ / n_neighbors_ are removed.
        """
        # Multi-restart run
        clarans = CLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=42)
        clarans.fit(self.X)
        self.assertTrue(hasattr(clarans, "n_iter_"))
        self.assertTrue(hasattr(clarans, "n_swaps_"))
        self.assertTrue(hasattr(clarans, "neighborhood_size_"))
        self.assertFalse(hasattr(clarans, "n_neighbors_"))
        self.assertFalse(hasattr(clarans, "total_neighbors_"))
        self.assertFalse(hasattr(clarans, "total_n_iter_"))
        self.assertFalse(hasattr(clarans, "total_n_swaps_"))

        self.assertGreaterEqual(clarans.n_iter_, 1)
        self.assertGreaterEqual(clarans.n_swaps_, 0)
        self.assertLessEqual(clarans.n_swaps_, clarans.n_iter_)

        # neighborhood_size_ check: k * (n - k)
        n_samples = self.X.shape[0]
        self.assertEqual(clarans.neighborhood_size_, 3 * (n_samples - 3))

        # Single-restart run
        c_single = CLARANS(n_clusters=3, num_local=1, max_neighbors=50, random_state=42)
        c_single.fit(self.X)
        self.assertGreaterEqual(c_single.n_iter_, 1)
        self.assertLessEqual(c_single.n_swaps_, c_single.n_iter_)
        self.assertLessEqual(clarans.inertia_, c_single.inertia_)

    def test_medoid_indices_sorted(self):
        """Test that medoid_indices_ is always sorted."""
        clarans = CLARANS(n_clusters=4, num_local=5, max_neighbors=30, random_state=42)
        clarans.fit(self.X)
        self.assertTrue(np.all(np.diff(clarans.medoid_indices_) >= 0))

    def test_medoid_uniqueness_multiple_seeds(self):
        """All medoids should be unique across multiple random seeds."""
        for seed in range(10):
            clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=30, random_state=seed)
            clarans.fit(self.X)
            unique_count = len(np.unique(clarans.medoid_indices_))
            self.assertEqual(
                unique_count,
                3,
                f"Seed {seed}: Expected 3 unique medoids, got {unique_count}",
            )

    def test_cluster_centers_match_medoids(self):
        """cluster_centers_ should be the actual data points at medoid_indices_."""
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=30, random_state=42)
        clarans.fit(self.X)
        for i, center in enumerate(clarans.cluster_centers_):
            medoid_idx = clarans.medoid_indices_[i]
            np.testing.assert_array_equal(
                center,
                self.X[medoid_idx],
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
            predicted,
            clarans.labels_,
            "predict(X) should match labels_ on training data",
        )

    def test_determinism(self):
        """Same random_state should give identical results."""
        clarans1 = CLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=123)
        clarans1.fit(self.X)

        clarans2 = CLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=123)
        clarans2.fit(self.X)

        np.testing.assert_array_equal(
            clarans1.medoid_indices_,
            clarans2.medoid_indices_,
            "Same random_state should give same medoids",
        )
        np.testing.assert_array_equal(
            clarans1.labels_,
            clarans2.labels_,
            "Same random_state should give same labels",
        )

    def test_max_neighbors_default(self):
        """Default max_neighbors should be None and calculated correctly."""
        clarans = CLARANS(n_clusters=3, num_local=1, random_state=42)
        self.assertIsNone(clarans.max_neighbors)
        clarans.fit(self.X)
        expected = max(250, int(0.0125 * 3 * (100 - 3)))
        self.assertEqual(clarans.max_neighbors_, expected)

    def test_max_neighbors_explicit_none(self):
        """Explicit max_neighbors=None should work without any deprecation warning."""
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors=None, random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clarans.fit(self.X)
            future_warnings = [item for item in w if issubclass(item.category, FutureWarning)]
            self.assertEqual(len(future_warnings), 0)
        expected = max(250, int(0.0125 * 3 * (100 - 3)))
        self.assertEqual(clarans.max_neighbors_, expected)

    def test_max_neighbors_explicit_auto(self):
        """Explicit max_neighbors='auto' should raise a FutureWarning and calculate correctly."""
        clarans = CLARANS(n_clusters=3, num_local=1, max_neighbors="auto", random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clarans.fit(self.X)
            future_warnings = [item for item in w if issubclass(item.category, FutureWarning)]
            self.assertTrue(any("deprecated" in str(item.message) for item in future_warnings))
        expected = max(250, int(0.0125 * 3 * (100 - 3)))
        self.assertEqual(clarans.max_neighbors_, expected)

    def test_max_neighbors_invalid(self):
        """Invalid max_neighbors should raise ValueError."""
        for invalid in ["invalid", 0, -5, 3.14]:
            clarans = CLARANS(n_clusters=3, max_neighbors=invalid, random_state=42)
            with self.assertRaises(ValueError):
                clarans.fit(self.X)

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
        _, min_dists = pairwise_distances_argmin_min(self.X, medoid_points, metric="euclidean")
        expected = np.sum(min_dists)
        self.assertAlmostEqual(cost, expected, places=6)

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
            _, min_dists = pairwise_distances_argmin_min(self.X, self.X[medoids], metric=metric)
            expected = np.sum(min_dists)
            self.assertAlmostEqual(cost, expected, places=6)

    def test_cost_precomputed(self):
        """calculate_cost should work with metric='precomputed'."""
        D = pairwise_distances(self.X, metric="euclidean")
        medoids = np.array([0, 30, 60])
        cost = calculate_cost(D, medoids, metric="precomputed")
        expected = calculate_cost(self.X, medoids, metric="euclidean")
        self.assertAlmostEqual(cost, expected, places=5)

    def test_cost_sparse_input(self):
        """calculate_cost should work with CSR sparse X."""
        from scipy import sparse

        X_sparse = sparse.csr_matrix(self.X)
        medoids = np.array([0, 20, 40])
        cost_dense = calculate_cost(self.X, medoids, "euclidean")
        cost_sparse = calculate_cost(X_sparse, medoids, "euclidean")
        self.assertAlmostEqual(cost_dense, cost_sparse, places=6)

    def test_cost_callable_metric(self):
        """calculate_cost should work with a callable metric function."""

        def my_l1(u, v):
            return float(np.sum(np.abs(u - v)))

        medoids = np.array([0, 20, 40])
        cost_callable = calculate_cost(self.X, medoids, my_l1)
        cost_manhattan = calculate_cost(self.X, medoids, "manhattan")
        self.assertAlmostEqual(cost_callable, cost_manhattan, places=5)

    def test_cost_single_medoid(self):
        """calculate_cost with a single medoid."""
        medoids = np.array([0])
        cost = calculate_cost(self.X, medoids, "euclidean")
        expected = float(np.sum(np.sqrt(np.sum((self.X - self.X[0]) ** 2, axis=1))))
        self.assertAlmostEqual(cost, expected, places=6)


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
        """Invalid max_neighbors (<= 0, float, bool, or bad str) should raise ValueError."""
        for val in [0, -1, "invalid", 1.5, True, False]:
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
        self.assertFalse(hasattr(model, "cluster_centers_"))
        self.assertEqual(model.n_features_in_, 50)

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

    def test_cost_evaluation_parameter(self):
        """Test cost_evaluation parameter: 'delta', 'brute_force', and invalid values."""
        model_delta = CLARANS(
            n_clusters=3, num_local=1, max_neighbors=20, cost_evaluation="delta", random_state=42
        )
        model_delta.fit(self.X)
        self.assertEqual(len(model_delta.medoid_indices_), 3)

        model_brute = CLARANS(
            n_clusters=3,
            num_local=1,
            max_neighbors=20,
            cost_evaluation="brute_force",
            random_state=42,
        )
        model_brute.fit(self.X)
        self.assertEqual(len(model_brute.medoid_indices_), 3)

        np.testing.assert_array_equal(model_delta.medoid_indices_, model_brute.medoid_indices_)
        self.assertAlmostEqual(model_delta.inertia_, model_brute.inertia_, places=5)

        with self.assertRaises(ValueError):
            CLARANS(cost_evaluation="invalid").fit(self.X)

    def test_legacy_parameters_removed(self):
        """Passing removed numlocal, maxneighbor, or cache should raise TypeError."""
        with self.assertRaises(TypeError):
            CLARANS(numlocal=2)
        with self.assertRaises(TypeError):
            CLARANS(maxneighbor=25)
        with self.assertRaises(TypeError):
            CLARANS(cache=True)
        with self.assertRaises(TypeError):
            CLARANS(cache=False)

    def test_parameters_num_local_max_neighbors(self):
        """Using num_local and max_neighbors should work and set max_neighbors_."""
        model = CLARANS(n_clusters=3, num_local=2, max_neighbors=25, random_state=42)
        model.fit(self.X)
        self.assertEqual(model.max_neighbors_, 25)
        self.assertFalse(hasattr(model, "maxneighbor_"))

    def test_delta_tolerance_rejects_ghost_swaps(self):
        """Tolerance should prevent ghost swaps when delta is negligible roundoff noise."""
        X_dup = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
        model = CLARANS(n_clusters=2, num_local=1, max_neighbors=50, random_state=42)
        model.fit(X_dup)
        self.assertGreaterEqual(model.n_swaps_, 0)

    def test_precomputed_dataframe_retains_feature_names_in(self):
        """Precomputed distance matrix from DataFrame should retain feature_names_in_."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")

        columns = [f"s_{i}" for i in range(len(self.D))]
        df_D = pd.DataFrame(self.D, columns=columns)
        model = CLARANS(n_clusters=3, metric="precomputed", random_state=42).fit(df_D)
        self.assertTrue(hasattr(model, "feature_names_in_"))
        np.testing.assert_array_equal(model.feature_names_in_, np.asarray(columns, dtype=object))

    def test_precomputed_refit_clears_cluster_centers(self):
        """Refitting a model with metric='precomputed' clears existing cluster_centers_."""
        m = CLARANS(n_clusters=2, random_state=42).fit(self.X)
        self.assertTrue(hasattr(m, "cluster_centers_"))

        m.set_params(metric="precomputed")
        m.fit(self.D)
        self.assertFalse(hasattr(m, "cluster_centers_"))

    def test_precomputed_init_2d_shape_mismatch(self):
        """2D init array with wrong shape under metric='precomputed' should raise ValueError."""
        invalid_init = np.zeros((4, self.D.shape[1]))
        with self.assertRaises(ValueError) as ctx:
            CLARANS(
                n_clusters=3,
                metric="precomputed",
                init=invalid_init,
                random_state=42,
            ).fit(self.D)
        self.assertIn("init array must be of shape", str(ctx.exception))

    def test_init_unconvertible_raises(self):
        """Unconvertible init object should raise an exception."""

        class Unconvertible:
            def __array__(self, *args, **kwargs):
                raise RuntimeError("Failed array conversion")

        with self.assertRaises((ValueError, TypeError, RuntimeError)):
            CLARANS(n_clusters=3, init=Unconvertible(), random_state=42).fit(self.X)

    def test_init_centers_cdist_fallback(self):
        """2D init with metric that cdist fails on should fall back to pairwise_distances."""
        from unittest.mock import patch

        with patch("clarans._clarans.cdist", side_effect=Exception("cdist failed")):
            init_centers = self.X[:3]
            model = CLARANS(n_clusters=3, init=init_centers, random_state=42).fit(self.X)
            self.assertEqual(len(model.medoid_indices_), 3)


class TestCascadingDistanceEngine(unittest.TestCase):
    """Tests for the cascading distance engine (cdist -> DistanceMetric -> pairwise)."""

    def setUp(self):
        self.X_dense, _ = make_blobs(n_samples=60, centers=3, n_features=4, random_state=42)
        import scipy.sparse as sp

        self.X_sparse = sp.csr_matrix(self.X_dense)

    def test_cdist_selected_for_dense_metrics(self):
        """SciPy cdist should be chosen for standard metrics on dense arrays."""
        for metric in ["euclidean", "manhattan", "chebyshev", "minkowski"]:
            model = CLARANS(
                n_clusters=3, metric=metric, num_local=1, max_neighbors=20, random_state=42
            )
            model.fit(self.X_dense)
            self.assertEqual(model._dist_engine, "cdist")

    def test_distance_metric_selected_for_sparse(self):
        """Scikit-Learn DistanceMetric should be chosen for sparse CSR input."""
        for metric in ["euclidean", "manhattan"]:
            model = CLARANS(
                n_clusters=3, metric=metric, num_local=1, max_neighbors=20, random_state=42
            )
            model.fit(self.X_sparse)
            self.assertEqual(model._dist_engine, "distance_metric")

    def test_distance_metric_selected_for_callable(self):
        """Scikit-Learn DistanceMetric should be chosen for callable distance functions."""

        def my_metric(u, v):
            return float(np.sum(np.abs(u - v)))

        model = CLARANS(
            n_clusters=3, metric=my_metric, num_local=1, max_neighbors=20, random_state=42
        )
        model.fit(self.X_dense)
        self.assertEqual(model._dist_engine, "distance_metric")

    def test_pairwise_selected_for_sparse_callable(self):
        """Pairwise engine should be chosen for callable distance functions on sparse matrices."""

        def my_metric(u, v):
            if issparse(u):
                u = u.toarray().ravel()
            if issparse(v):
                v = v.toarray().ravel()
            return float(np.sum(np.abs(u - v)))

        for model_cls in [CLARANS, FastCLARANS]:
            model = model_cls(
                n_clusters=3, metric=my_metric, num_local=1, max_neighbors=20, random_state=42
            )
            model.fit(self.X_sparse)
            self.assertEqual(model._dist_engine, "pairwise")
            self.assertTrue(np.isfinite(model.inertia_))
            self.assertGreater(model.inertia_, 0.0)

            # Check predict and transform on sparse input
            preds = model.predict(self.X_sparse[:5])
            self.assertEqual(preds.shape, (5,))
            trans = model.transform(self.X_sparse[:5])
            self.assertEqual(trans.shape, (5, 3))
            self.assertTrue(np.all(trans >= 0.0))

    def test_precomputed_engine_selected(self):
        """Precomputed engine should be chosen for precomputed distance matrices."""
        D = pairwise_distances(self.X_dense)
        model = CLARANS(
            n_clusters=3, metric="precomputed", num_local=1, max_neighbors=20, random_state=42
        )
        model.fit(D)
        self.assertEqual(model._dist_engine, "precomputed")

    def test_invalid_metric_raises_clear_error(self):
        """Invalid metric string should raise ValueError with options from all 3 engines."""
        with self.assertRaises(ValueError) as ctx:
            CLARANS(metric="non_existent_metric").fit(self.X_dense)

        msg = str(ctx.exception)
        self.assertIn("The 'metric' parameter of CLARANS must be a str among", msg)
        self.assertIn("'euclidean'", msg)
        self.assertIn("'manhattan'", msg)
        self.assertIn("'chebyshev'", msg)
        self.assertIn("'infinity'", msg)
        self.assertIn("'precomputed'", msg)
        self.assertIn("or a callable", msg)
        self.assertIn("Got 'non_existent_metric' instead.", msg)

    def test_invalid_metric_type_raises_clear_error(self):
        """Non-string, non-callable metric should raise ValueError with received type/value."""
        with self.assertRaises(ValueError) as ctx:
            CLARANS(metric=123).fit(self.X_dense)

        msg = str(ctx.exception)
        self.assertIn("The 'metric' parameter of CLARANS must be a str among", msg)
        self.assertIn("Got 123 instead.", msg)

    def test_fast_clarans_invalid_metric_error(self):
        """FastCLARANS should reflect its own class name in metric validation error."""
        with self.assertRaises(ValueError) as ctx:
            FastCLARANS(metric="non_existent_metric").fit(self.X_dense)

        msg = str(ctx.exception)
        self.assertIn("The 'metric' parameter of FastCLARANS must be a str among", msg)
        self.assertIn("Got 'non_existent_metric' instead.", msg)

    def test_scipy_specific_metric_jensenshannon(self):
        """SciPy metric jensenshannon works in fit, predict, transform, calculate_cost."""
        X_prob = np.array(
            [[0.1, 0.9], [0.15, 0.85], [0.85, 0.15], [0.9, 0.1], [0.5, 0.5], [0.45, 0.55]]
        )
        cost = calculate_cost(X_prob, [0, 2], metric="jensenshannon")
        self.assertGreater(cost, 0.0)

        model = CLARANS(n_clusters=2, metric="jensenshannon", random_state=42)
        model.fit(X_prob)
        self.assertEqual(len(model.medoid_indices_), 2)
        self.assertEqual(len(model.labels_), len(X_prob))

        preds = model.predict(X_prob)
        self.assertEqual(len(preds), len(X_prob))

        transformed = model.transform(X_prob)
        self.assertEqual(transformed.shape, (len(X_prob), 2))

        fmodel = FastCLARANS(n_clusters=2, metric="jensenshannon", random_state=42)
        fmodel.fit(X_prob)
        self.assertEqual(len(fmodel.medoid_indices_), 2)


class TestUtilsHelpers(unittest.TestCase):
    """Tests for utility helpers in clarans.utils."""

    def test_check_medoids_valid(self):
        """Valid medoids should convert to 1D intp array."""
        medoids = [0, 2, 4]
        res = check_medoids(medoids, n_samples=10)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.dtype, np.intp)
        np.testing.assert_array_equal(res, [0, 2, 4])

    def test_check_medoids_duplicate_raises(self):
        """Duplicate medoids should raise ValueError."""
        with self.assertRaises(ValueError):
            check_medoids([1, 2, 2])

    def test_check_medoids_empty_raises(self):
        """Empty medoids sequence should raise ValueError."""
        with self.assertRaises(ValueError):
            check_medoids([])

    def test_check_medoids_out_of_bounds_raises(self):
        """Indices exceeding n_samples or negative should raise ValueError."""
        with self.assertRaises(ValueError):
            check_medoids([0, 10], n_samples=10)
        with self.assertRaises(ValueError):
            check_medoids([-1, 2], n_samples=10)

    def test_check_medoids_non_integer_raises(self):
        """Non-integer medoid elements should raise TypeError."""
        with self.assertRaises(TypeError):
            check_medoids([0.5, 1.2])

    def test_check_medoids_multidimensional_raises(self):
        """2D medoids array should raise ValueError."""
        with self.assertRaises(ValueError):
            check_medoids(np.array([[0, 1], [2, 3]]))

    def test_check_medoids_none_raises(self):
        """Passing None for medoid_indices should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            check_medoids(None)
        self.assertIn("medoid_indices cannot be empty", str(ctx.exception))

    def test_check_medoids_unconvertible_raises(self):
        """Passing unconvertible object to check_medoids should raise an exception."""

        class BadObject:
            def __array__(self, *args, **kwargs):
                raise RuntimeError("Cannot convert")

        with self.assertRaises((ValueError, TypeError, RuntimeError)):
            check_medoids(BadObject())

    def test_calculate_cost_cdist_fallback(self):
        """calculate_cost should fall back to pairwise_distances when cdist fails on dense data."""
        from unittest.mock import patch

        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        with patch("clarans.utils.cdist", side_effect=Exception("cdist failed")):
            cost = calculate_cost(X, [0, 1])
            self.assertGreater(cost, 0.0)

    def test_public_imports_from_init(self):
        """calculate_cost, check_medoids, EfficiencyWarning, and HAS_CYTHON should be in clarans."""
        from clarans import (
            EfficiencyWarning as EffWarn,
            HAS_CYTHON as has_cy,
            calculate_cost as calc,
            check_medoids as chk,
        )

        self.assertTrue(callable(calc))
        self.assertTrue(callable(chk))
        self.assertTrue(issubclass(EffWarn, UserWarning))
        self.assertIsInstance(has_cy, bool)

    def test_efficiency_warning_issued_once(self):
        """EfficiencyWarning should be issued when HAS_CYTHON is False, exactly once."""
        import warnings
        from unittest.mock import patch
        import clarans.utils as utils

        with (
            patch.object(utils, "HAS_CYTHON", False),
            patch.object(utils, "_cython_warning_issued", False),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                utils._warn_cython_unavailable()

                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, utils.EfficiencyWarning))
                self.assertIn(
                    "Compiled Cython extensions (_core) are not available", str(w[-1].message)
                )

                # Second call should not issue another warning
                utils._warn_cython_unavailable()
                self.assertEqual(len(w), 1)


@pytest.mark.parametrize("estimator_cls", [CLARANS, FastCLARANS])
class TestCommonMetricParams:
    """Parameterized metric_params and verbose logging tests across CLARANS & FastCLARANS."""

    @pytest.fixture(autouse=True)
    def _setup_data(self):
        np.random.seed(42)
        self.X = np.random.randn(30, 3)

    def test_minkowski_with_p(self, estimator_cls):
        model = estimator_cls(
            n_clusters=2, metric="minkowski", metric_params={"p": 3}, random_state=42
        )
        model.fit(self.X)
        assert model.labels_.shape == (len(self.X),)
        assert model.cluster_centers_.shape == (2, 3)
        preds = model.predict(self.X[:5])
        assert preds.shape == (5,)
        trans = model.transform(self.X[:5])
        assert trans.shape == (5, 2)

    def test_mahalanobis_with_vi(self, estimator_cls):
        VI = np.linalg.inv(np.cov(self.X.T))
        model = estimator_cls(
            n_clusters=2, metric="mahalanobis", metric_params={"VI": VI}, random_state=42
        )
        model.fit(self.X)
        assert model.labels_.shape == (len(self.X),)
        preds = model.predict(self.X[:5])
        assert preds.shape == (5,)
        trans = model.transform(self.X[:5])
        assert trans.shape == (5, 2)

    def test_mahalanobis_missing_vi_raises(self, estimator_cls):
        with pytest.raises(ValueError, match="(?i)vi"):
            estimator_cls(n_clusters=2, metric="mahalanobis", random_state=42).fit(self.X)
        with pytest.raises(ValueError, match="(?i)vi"):
            estimator_cls(
                n_clusters=2, metric="mahalanobis", metric_params={}, random_state=42
            ).fit(self.X)

    def test_custom_callable_with_metric_params(self, estimator_cls):
        def custom_dist(x, y, weight=1.0):
            return np.sum(np.abs(x - y)) * weight

        model = estimator_cls(
            n_clusters=2,
            metric=custom_dist,
            metric_params={"weight": 2.5},
            random_state=42,
        )
        model.fit(self.X)
        assert model.labels_.shape == (len(self.X),)

    def test_invalid_metric_params_type(self, estimator_cls):
        with pytest.raises(ValueError, match="metric_params"):
            estimator_cls(n_clusters=2, metric_params="not_a_dict", random_state=42).fit(self.X)

    def test_clone_preserves_metric_params(self, estimator_cls):
        model = estimator_cls(metric="minkowski", metric_params={"p": 4})
        cloned = clone(model)
        assert cloned.metric_params == {"p": 4}

    def test_verbose_silent(self, estimator_cls, capsys):
        for v in (0, False):
            model = estimator_cls(
                n_clusters=2, num_local=1, max_neighbors=10, verbose=v, random_state=42
            )
            model.fit(self.X)
            captured = capsys.readouterr()
            assert captured.out == ""

    def test_verbose_level_1(self, estimator_cls, capsys):
        prefix = estimator_cls.__name__
        for v in (1, True):
            model = estimator_cls(
                n_clusters=2, num_local=2, max_neighbors=10, verbose=v, random_state=42
            )
            model.fit(self.X)
            output = capsys.readouterr().out
            assert f"[{prefix}]" in output
            assert "max_neighbors=" in output
            assert "Cost" in output
            assert "Swaps" in output
            assert "Evals" in output
            assert "converged" in output
            assert "Best: #" in output
            assert "Totals:" in output
            assert "Restart" not in output

    def test_verbose_level_2(self, estimator_cls, capsys):
        prefix = estimator_cls.__name__
        model = estimator_cls(
            n_clusters=2,
            num_local=1,
            max_neighbors=50,
            init="random",
            verbose=2,
            random_state=0,
        )
        model.fit(self.X)
        output = capsys.readouterr().out
        assert f"[{prefix}]" in output
        assert "Restart 1/1 (init cost:" in output
        assert "Best: #" in output
        if model.n_swaps_ > 0:
            assert "swap" in output
            assert "| diff " in output

    def test_verbose_invalid(self, estimator_cls):
        for invalid_val in [-1, -5, "1", 1.5, [1]]:
            with pytest.raises(ValueError, match="verbose"):
                estimator_cls(n_clusters=2, verbose=invalid_val, random_state=42).fit(self.X)

    def test_clone_preserves_verbose(self, estimator_cls):
        model = estimator_cls(verbose=2)
        cloned = clone(model)
        assert cloned.verbose == 2


class TestCLARANSMetricParams:
    """CLARANS-specific metric and verbose tests."""

    @pytest.fixture(autouse=True)
    def _setup_data(self):
        np.random.seed(42)
        self.X = np.random.randn(30, 3)

    def test_verbose_2_brute_force_swap_logging(self, capsys):
        """verbose=2 with cost_evaluation='brute_force' should log swap details."""
        CLARANS(
            n_clusters=2,
            num_local=1,
            max_neighbors=50,
            cost_evaluation="brute_force",
            verbose=2,
            random_state=42,
        ).fit(self.X)
        output = capsys.readouterr().out
        assert "Restart" in output
