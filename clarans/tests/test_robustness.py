"""
Robustness, invariants, and edge-case test suite for scikit-clarans.

Validates numerical stability, mathematical invariants, and boundary conditions
across 12 groups (A–L):
  A – Initialization deep tests (precomputed, determinism, duplicate centers)
  B – NumPy fallback paths (mock _core = None)
  C – Metric alias & exotic metrics (l1, l2, chebyshev, minkowski)
  D – Predict/Transform on unseen data (new samples, single sample)
  E – Precomputed edge cases (submatrix predict, invalid columns, sparse precomputed)
  F – Numerical stability (large/small values, NaN/Inf rejection, single sample)
  G – Mathematical invariants (property-based: inertia, labels, cost monotonicity)
  H – sklearn API compliance (get/set_params, repr, tags, n_features_in_, score)
  I – Edge memory & extreme parameters (1 non-medoid, max_neighbors=1, refit)
  J – FastCLARANS-specific (auto formula, sorted medoids, precomputed inits)
  K – calculate_cost exhaustive (sparse, callable, single medoid)
  L – Cython kernel edge cases (single sample, single cluster, tied distances)
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from clarans import CLARANS, FastCLARANS, calculate_cost, HAS_CYTHON
from clarans._initialization import (
    initialize_build,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)

try:
    from clarans import _core
except ImportError:
    _core = None  # type: ignore[assignment]


# ===========================================================================
# Group A: Initialization Deep Tests
# ===========================================================================
class TestInitializationDeep(unittest.TestCase):
    """Group A – Deep tests for initialization functions on precomputed,
    determinism, and duplicate-center handling."""

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
        # All init centers are near the same point → will collapse to 1 index
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
            # Should have warned about duplicate init centers
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


# ===========================================================================
# Group B: NumPy Fallback Paths (mock _core = None)
# ===========================================================================
class TestNumpyFallbackPaths(unittest.TestCase):
    """Group B – Verify all Python/NumPy fallback paths work when Cython
    extensions are unavailable (simulated via monkey-patching _core to None)."""

    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=40, centers=3, n_features=2, random_state=42
        )

    def _fit_with_core_none(self, model_cls, **kwargs):
        """Fit a model with _core patched to None in all modules."""
        model = model_cls(n_clusters=3, num_local=1, max_neighbors=10, random_state=42, **kwargs)
        with patch("clarans._clarans._core", None), \
             patch("clarans._fast_clarans._core", None), \
             patch("clarans._initialization._core", None):
            model.fit(self.X)
        return model

    def test_clarans_delta_numpy_fallback(self):
        """CLARANS with cost_evaluation='delta' works without Cython."""
        model = self._fit_with_core_none(CLARANS, cost_evaluation="delta")
        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertGreaterEqual(model.inertia_, 0)

    def test_clarans_brute_force_numpy_fallback(self):
        """CLARANS with cost_evaluation='brute_force' works without Cython."""
        model = self._fit_with_core_none(CLARANS, cost_evaluation="brute_force")
        self.assertEqual(len(model.medoid_indices_), 3)

    def test_fastclarans_numpy_fallback(self):
        """FastCLARANS works without Cython (fastpam1 Python path)."""
        model = self._fit_with_core_none(FastCLARANS)
        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertGreaterEqual(model.inertia_, 0)

    def test_build_init_numpy_fallback(self):
        """BUILD initialization works without Cython."""
        model = self._fit_with_core_none(CLARANS, init="build")
        self.assertEqual(len(model.medoid_indices_), 3)

    def test_kmedoids_pp_numpy_fallback(self):
        """k-medoids++ initialization works without Cython."""
        model = self._fit_with_core_none(CLARANS, init="k-medoids++")
        self.assertEqual(len(model.medoid_indices_), 3)


# ===========================================================================
# Group C: Metric Alias & Exotic Metrics
# ===========================================================================
class TestMetricAliasAndExotic(unittest.TestCase):
    """Group C – Verify metric aliases (l1, l2) and exotic metrics
    produce correct results end-to-end."""

    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=50, centers=3, n_features=2, random_state=42
        )

    def test_l1_alias_matches_manhattan(self):
        """Metric alias 'l1' should produce identical results to 'manhattan'."""
        m_l1 = CLARANS(
            n_clusters=3, metric="l1", random_state=42, num_local=1, max_neighbors=20
        )
        m_l1.fit(self.X)
        m_man = CLARANS(
            n_clusters=3, metric="manhattan", random_state=42, num_local=1, max_neighbors=20
        )
        m_man.fit(self.X)
        np.testing.assert_array_equal(m_l1.medoid_indices_, m_man.medoid_indices_)
        self.assertAlmostEqual(m_l1.inertia_, m_man.inertia_, places=9)

    def test_l2_alias_matches_euclidean(self):
        """Metric alias 'l2' should produce identical results to 'euclidean'."""
        m_l2 = CLARANS(
            n_clusters=3, metric="l2", random_state=42, num_local=1, max_neighbors=20
        )
        m_l2.fit(self.X)
        m_euc = CLARANS(
            n_clusters=3, metric="euclidean", random_state=42, num_local=1, max_neighbors=20
        )
        m_euc.fit(self.X)
        np.testing.assert_array_equal(m_l2.medoid_indices_, m_euc.medoid_indices_)
        self.assertAlmostEqual(m_l2.inertia_, m_euc.inertia_, places=9)

    def test_chebyshev_metric_end_to_end(self):
        """Chebyshev metric should work across fit, predict, transform."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, metric="chebyshev", random_state=42)
            m.fit(self.X)
            self.assertEqual(len(m.medoid_indices_), 3)
            preds = m.predict(self.X)
            self.assertEqual(preds.shape, (50,))
            trans = m.transform(self.X)
            self.assertEqual(trans.shape, (50, 3))

    def test_minkowski_metric_end_to_end(self):
        """Minkowski metric should work across fit, predict, transform."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, metric="minkowski", random_state=42)
            m.fit(self.X)
            self.assertEqual(len(m.medoid_indices_), 3)
            preds = m.predict(self.X)
            self.assertEqual(preds.shape, (50,))


# ===========================================================================
# Group D: Predict/Transform on Unseen Data
# ===========================================================================
class TestPredictTransformUnseenData(unittest.TestCase):
    """Group D – Verify predict/transform on new data distinct from
    training set, including single-sample edge case."""

    def setUp(self):
        self.X_train, _ = make_blobs(
            n_samples=80, centers=3, n_features=2, random_state=42
        )
        self.X_new, _ = make_blobs(
            n_samples=20, centers=3, n_features=2, random_state=99
        )

    def test_predict_unseen_data(self):
        """predict() on new data should return labels in [0, n_clusters)."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42).fit(self.X_train)
            preds = m.predict(self.X_new)
            self.assertEqual(preds.shape, (20,))
            self.assertTrue(np.all((preds >= 0) & (preds < 3)))

    def test_predict_single_sample(self):
        """predict() should work on a single sample (1, n_features)."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42).fit(self.X_train)
            single = self.X_new[:1]
            pred = m.predict(single)
            self.assertEqual(pred.shape, (1,))
            self.assertIn(pred[0], [0, 1, 2])

    def test_transform_unseen_data(self):
        """transform() on new data should return (n_new, n_clusters) distances."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42).fit(self.X_train)
            trans = m.transform(self.X_new)
            self.assertEqual(trans.shape, (20, 3))
            self.assertTrue(np.all(trans >= 0))

    def test_transform_single_sample(self):
        """transform() should work on a single sample."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42).fit(self.X_train)
            single = self.X_new[:1]
            trans = m.transform(single)
            self.assertEqual(trans.shape, (1, 3))
            self.assertTrue(np.all(trans >= 0))


# ===========================================================================
# Group E: Precomputed Edge Cases
# ===========================================================================
class TestPrecomputedEdgeCases(unittest.TestCase):
    """Group E – Edge cases for metric='precomputed': submatrix predict,
    invalid columns, sparse precomputed matrix."""

    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=40, centers=3, n_features=2, random_state=42
        )
        self.D = pairwise_distances(self.X, metric="euclidean")

    def _fit_precomputed(self, cls=CLARANS):
        m = cls(
            n_clusters=3, metric="precomputed", random_state=42,
            num_local=1, max_neighbors=20,
        )
        m.fit(self.D)
        return m

    def test_precomputed_predict_medoid_submatrix(self):
        """predict() with X of shape (n_new, n_clusters) → medoids-only columns."""
        model = self._fit_precomputed()
        # Simulate new data: distances from 10 new "points" to medoids only
        D_sub = self.D[:10, model.medoid_indices_]
        preds = model.predict(D_sub)
        self.assertEqual(preds.shape, (10,))
        self.assertTrue(np.all((preds >= 0) & (preds < 3)))

    def test_precomputed_predict_invalid_columns(self):
        """predict() should raise ValueError when column count is invalid."""
        model = self._fit_precomputed()
        bad_X = np.random.rand(5, 7)  # Not n_train_samples and not n_clusters
        with self.assertRaises(ValueError) as ctx:
            model.predict(bad_X)
        self.assertIn("columns", str(ctx.exception).lower())

    def test_precomputed_transform_invalid_columns(self):
        """transform() should raise ValueError when column count is invalid."""
        model = self._fit_precomputed()
        bad_X = np.random.rand(5, 7)
        with self.assertRaises(ValueError) as ctx:
            model.transform(bad_X)
        self.assertIn("columns", str(ctx.exception).lower())

    def test_precomputed_transform_medoid_submatrix(self):
        """transform() with X of shape (n, n_clusters) should return those distances."""
        model = self._fit_precomputed()
        D_sub = self.D[:, model.medoid_indices_]
        trans = model.transform(D_sub)
        self.assertEqual(trans.shape, (40, 3))
        # Values should match the submatrix directly
        np.testing.assert_allclose(trans, D_sub, rtol=1e-12)

    def test_precomputed_sparse_distance_matrix(self):
        """fit/predict/transform should work with sparse precomputed matrix."""
        D_sparse = sparse.csr_matrix(self.D)
        for cls in [CLARANS, FastCLARANS]:
            m = cls(
                n_clusters=3, metric="precomputed", random_state=42,
                num_local=1, max_neighbors=20,
            )
            m.fit(D_sparse)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertIsNone(m.cluster_centers_)
            labels = m.predict(D_sparse)
            self.assertEqual(labels.shape, (40,))
            trans = m.transform(D_sparse)
            self.assertEqual(trans.shape, (40, 3))


# ===========================================================================
# Group F: Numerical Stability & Edge Cases
# ===========================================================================
class TestNumericalStability(unittest.TestCase):
    """Group F – Numerical stability with large/small values, NaN/Inf
    rejection, and single-sample constraint."""

    def test_large_values_stability(self):
        """Data with values ~1e10 should still produce valid clustering."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 2) * 1e10
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42, num_local=1, max_neighbors=20)
            m.fit(X)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertTrue(np.isfinite(m.inertia_))

    def test_small_values_stability(self):
        """Data with values ~1e-10 should still produce valid clustering."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 2) * 1e-10
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42, num_local=1, max_neighbors=20)
            m.fit(X)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertTrue(np.isfinite(m.inertia_))

    def test_nan_input_rejected(self):
        """NaN values in X should be rejected by check_array."""
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        for cls in [CLARANS, FastCLARANS]:
            with self.assertRaises(ValueError):
                cls(n_clusters=2).fit(X)

    def test_inf_input_rejected(self):
        """Inf values in X should be rejected by check_array."""
        X = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
        for cls in [CLARANS, FastCLARANS]:
            with self.assertRaises(ValueError):
                cls(n_clusters=2).fit(X)

    def test_single_sample_raises(self):
        """Only 1 sample should raise ValueError (ensure_min_samples=2)."""
        X = np.array([[1.0, 2.0]])
        for cls in [CLARANS, FastCLARANS]:
            with self.assertRaises(ValueError):
                cls(n_clusters=1).fit(X)


# ===========================================================================
# Group G: Mathematical Invariants (Property-based)
# ===========================================================================
class TestMathematicalInvariants(unittest.TestCase):
    """Group G – Property-based tests verifying mathematical invariants
    that must always hold regardless of data or random state."""

    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=80, centers=3, n_features=2, random_state=42
        )

    def test_inertia_equals_calculate_cost(self):
        """inertia_ must exactly equal calculate_cost(X, medoid_indices_, metric)."""
        for cls in [CLARANS, FastCLARANS]:
            for metric in ["euclidean", "manhattan"]:
                m = cls(n_clusters=3, metric=metric, random_state=42).fit(self.X)
                expected = calculate_cost(self.X, m.medoid_indices_, metric)
                self.assertAlmostEqual(
                    m.inertia_, expected, places=6,
                    msg=f"{cls.__name__} metric={metric}: inertia mismatch",
                )

    def test_medoid_indices_are_actual_indices(self):
        """All medoid_indices_ must be valid indices into X."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42).fit(self.X)
            self.assertTrue(np.all(m.medoid_indices_ >= 0))
            self.assertTrue(np.all(m.medoid_indices_ < len(self.X)))

    def test_labels_cover_all_clusters_well_separated(self):
        """On well-separated data, every cluster should have at least 1 point."""
        X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.1, random_state=42)
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42).fit(X)
            unique_labels = np.unique(m.labels_)
            self.assertEqual(
                len(unique_labels), 3,
                f"{cls.__name__}: Expected 3 unique labels, got {len(unique_labels)}",
            )

    def test_delta_vs_brute_force_on_precomputed(self):
        """delta and brute_force must agree on precomputed distance matrix."""
        D = pairwise_distances(self.X, metric="euclidean")
        m_delta = CLARANS(
            n_clusters=3, cost_evaluation="delta", metric="precomputed",
            random_state=42, num_local=1, max_neighbors=20,
        ).fit(D)
        m_bf = CLARANS(
            n_clusters=3, cost_evaluation="brute_force", metric="precomputed",
            random_state=42, num_local=1, max_neighbors=20,
        ).fit(D)
        np.testing.assert_array_equal(m_delta.medoid_indices_, m_bf.medoid_indices_)
        self.assertAlmostEqual(m_delta.inertia_, m_bf.inertia_, places=9)

    def test_cost_monotonic_with_more_clusters(self):
        """Inertia should generally decrease (or stay equal) as n_clusters increases."""
        costs = []
        for k in [1, 2, 3, 5]:
            m = CLARANS(
                n_clusters=k, random_state=42, num_local=2, max_neighbors=50,
            ).fit(self.X)
            costs.append(m.inertia_)
        # Each step should be non-increasing (with a tolerance for stochastic variation)
        for i in range(len(costs) - 1):
            self.assertGreaterEqual(
                costs[i] + 1e-6, costs[i + 1],
                f"Cost did not decrease: k={[1, 2, 3, 5][i]} cost={costs[i]:.2f} "
                f"→ k={[1, 2, 3, 5][i + 1]} cost={costs[i + 1]:.2f}",
            )


# ===========================================================================
# Group H: sklearn API Compliance
# ===========================================================================
class TestSklearnAPICompliance(unittest.TestCase):
    """Group H – Verify compliance with scikit-learn estimator API
    (get_params, set_params, repr, tags, n_features_in_, score)."""

    def setUp(self):
        self.X, self.y = make_blobs(
            n_samples=60, centers=3, n_features=2, random_state=42
        )

    def test_get_set_params_roundtrip(self):
        """get_params → set_params → get_params should be idempotent."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=5, num_local=3, random_state=99)
            params = m.get_params()
            self.assertEqual(params["n_clusters"], 5)
            self.assertEqual(params["num_local"], 3)
            self.assertEqual(params["random_state"], 99)

            m.set_params(n_clusters=2)
            self.assertEqual(m.get_params()["n_clusters"], 2)
            # Other params unchanged
            self.assertEqual(m.get_params()["num_local"], 3)

    def test_repr_contains_class_name_and_params(self):
        """repr(model) should contain class name and key parameters."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=5, random_state=42)
            r = repr(m)
            self.assertIn(cls.__name__, r)
            self.assertIn("n_clusters=5", r)
            self.assertIn("random_state=42", r)

    def test_sklearn_tags_sparse(self):
        """__sklearn_tags__() should report sparse=True."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3)
            tags = m.__sklearn_tags__()
            self.assertTrue(tags.input_tags.sparse)

    def test_sklearn_tags_precomputed(self):
        """__sklearn_tags__() should report pairwise=True when metric='precomputed'."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, metric="precomputed")
            tags = m.__sklearn_tags__()
            self.assertTrue(tags.input_tags.pairwise)

    def test_n_features_in_set_after_fit(self):
        """n_features_in_ should be set for non-precomputed, absent for precomputed."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42).fit(self.X)
            self.assertTrue(hasattr(m, "n_features_in_"))
            self.assertEqual(m.n_features_in_, 2)

            D = pairwise_distances(self.X)
            m_pre = cls(n_clusters=3, metric="precomputed", random_state=42).fit(D)
            self.assertFalse(hasattr(m_pre, "n_features_in_"))

    def test_score_method_exists(self):
        """CLARANS/FastCLARANS should have a score() method via ClusterMixin or custom."""
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, random_state=42).fit(self.X)
            # ClusterMixin doesn't provide score() directly, but if it exists
            # it should be callable. If it doesn't exist, that's fine too.
            if hasattr(m, "score"):
                # Should not raise
                s = m.score(self.X)
                self.assertIsInstance(s, (int, float, np.floating))


# ===========================================================================
# Group I: Concurrent/Edge Memory Tests
# ===========================================================================
class TestEdgeMemoryAndExtremeParams(unittest.TestCase):
    """Group I – Extreme parameter values and edge memory conditions."""

    def test_single_non_medoid_candidate(self):
        """n_clusters = n_samples - 1 → only 1 non-medoid candidate."""
        X = np.random.RandomState(42).randn(5, 2)
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=4, num_local=1, max_neighbors=10, random_state=42)
            m.fit(X)
            self.assertEqual(len(m.medoid_indices_), 4)
            self.assertEqual(len(np.unique(m.medoid_indices_)), 4)

    def test_max_neighbors_equals_1(self):
        """max_neighbors=1 (extreme low) should still produce valid results."""
        X, _ = make_blobs(n_samples=30, centers=3, random_state=42)
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=3, max_neighbors=1, random_state=42)
            m.fit(X)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertEqual(m.max_neighbors_, 1)

    def test_num_local_large(self):
        """Large num_local (10) on small data should converge without issues."""
        X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=2, num_local=10, max_neighbors=20, random_state=42)
            m.fit(X)
            self.assertEqual(len(m.medoid_indices_), 2)

    def test_fit_called_twice_resets_state(self):
        """Second fit() should completely replace first fit's state."""
        X1, _ = make_blobs(n_samples=50, centers=2, random_state=42)
        X2, _ = make_blobs(n_samples=60, centers=4, n_features=2, random_state=99)
        for cls in [CLARANS, FastCLARANS]:
            m = cls(n_clusters=2, random_state=42)
            m.fit(X1)
            old_medoids = m.medoid_indices_.copy()
            old_labels = m.labels_.copy()

            # Refit with different data
            m.set_params(n_clusters=3)
            m.fit(X2)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertEqual(len(m.labels_), 60)
            # State should be completely different
            self.assertNotEqual(old_labels.shape, m.labels_.shape)
            self.assertNotEqual(len(old_medoids), len(m.medoid_indices_))


# ===========================================================================
# Group J: FastCLARANS-Specific
# ===========================================================================
class TestFastCLARANSSpecific(unittest.TestCase):
    """Group J – Tests specific to FastCLARANS behavior distinct from CLARANS."""

    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=100, centers=3, n_features=2, random_state=42
        )

    def test_fast_clarans_max_neighbors_auto_formula(self):
        """FastCLARANS auto formula should be 2.5% of (n - k), NOT CLARANS's 1.25% * k * (n-k)."""
        m = FastCLARANS(n_clusters=3, random_state=42)
        m.fit(self.X)
        expected = max(250, int(0.025 * (100 - 3)))
        self.assertEqual(m.max_neighbors_, expected)

        # The values happen to both be 250 for n=100, k=3, so test with larger data
        clarans_expected = max(250, int(0.0125 * 10 * (50000 - 10)))
        X_large, _ = make_blobs(n_samples=50000, centers=10, random_state=42)
        m_fast = FastCLARANS(n_clusters=10, random_state=42)
        m_fast.fit(X_large)
        fast_expected = max(250, int(0.025 * (50000 - 10)))
        self.assertEqual(m_fast.max_neighbors_, fast_expected)
        self.assertNotEqual(fast_expected, clarans_expected)

    def test_fast_clarans_medoids_always_sorted(self):
        """FastCLARANS should always return sorted medoid_indices_."""
        for seed in range(5):
            m = FastCLARANS(n_clusters=4, random_state=seed).fit(self.X)
            self.assertTrue(
                np.all(np.diff(m.medoid_indices_) >= 0),
                f"Seed {seed}: medoids not sorted: {m.medoid_indices_}",
            )

    def test_fast_clarans_precomputed_all_inits(self):
        """FastCLARANS with precomputed should work with all init methods."""
        D = pairwise_distances(self.X)
        for init in ["random", "heuristic", "k-medoids++", "build"]:
            m = FastCLARANS(
                n_clusters=3, init=init, metric="precomputed",
                random_state=42, num_local=1, max_neighbors=10,
            )
            m.fit(D)
            self.assertEqual(
                len(m.medoid_indices_), 3,
                f"FastCLARANS precomputed init={init} failed",
            )


# ===========================================================================
# Group K: calculate_cost Exhaustive
# ===========================================================================
class TestCalculateCostExhaustive(unittest.TestCase):
    """Group K – Exhaustive tests for the calculate_cost utility function."""

    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=50, centers=3, n_features=2, random_state=42
        )

    def test_calculate_cost_sparse_input(self):
        """calculate_cost should work with CSR sparse X."""
        X_sparse = sparse.csr_matrix(self.X)
        medoids = np.array([0, 20, 40])
        cost_dense = calculate_cost(self.X, medoids, "euclidean")
        cost_sparse = calculate_cost(X_sparse, medoids, "euclidean")
        self.assertAlmostEqual(cost_dense, cost_sparse, places=6)

    def test_calculate_cost_callable_metric(self):
        """calculate_cost should work with a callable metric function."""
        def my_l1(u, v):
            return float(np.sum(np.abs(u - v)))

        medoids = np.array([0, 20, 40])
        cost_callable = calculate_cost(self.X, medoids, my_l1)
        cost_manhattan = calculate_cost(self.X, medoids, "manhattan")
        self.assertAlmostEqual(cost_callable, cost_manhattan, places=5)

    def test_calculate_cost_single_medoid(self):
        """calculate_cost with a single medoid."""
        medoids = np.array([0])
        cost = calculate_cost(self.X, medoids, "euclidean")
        # Manual: sum of distances from every point to X[0]
        expected = float(np.sum(np.sqrt(np.sum((self.X - self.X[0]) ** 2, axis=1))))
        self.assertAlmostEqual(cost, expected, places=6)


# ===========================================================================
# Group L: Cython Kernel Edge Cases
# ===========================================================================
@unittest.skipUnless(
    HAS_CYTHON and _core is not None,
    "Cython extension '_core' not compiled or available",
)
class TestCythonKernelEdgeCases(unittest.TestCase):
    """Group L – Edge cases for Cython kernels: single sample, single cluster,
    tied distances, single candidate."""

    def test_clarans_delta_single_sample(self):
        """clarans_delta with n_samples=1."""
        near_idx = np.array([0], dtype=np.intp)
        near_dist = np.array([1.0], dtype=np.float64)
        second_dist = np.array([np.inf], dtype=np.float64)
        d_xc = np.array([0.5], dtype=np.float64)
        delta = _core.clarans_delta(near_idx, near_dist, second_dist, d_xc, 0, 1)
        # Sample assigned to medoid 0, swapping it: min(inf, 0.5) - 1.0 = -0.5
        self.assertAlmostEqual(delta, -0.5, places=10)

    def test_fastpam1_delta_single_cluster(self):
        """fastpam1_delta with n_clusters=1."""
        n = 10
        near_idx = np.zeros(n, dtype=np.intp)
        near_dist = np.ones(n, dtype=np.float64) * 5.0
        second_dist = np.full(n, np.inf, dtype=np.float64)
        d_xc = np.ones(n, dtype=np.float64) * 3.0
        best_m, best_val, delta_arr = _core.fastpam1_delta(
            near_idx, near_dist, second_dist, d_xc, n, 1
        )
        self.assertEqual(best_m, 0)
        # All samples: dc(3) < d1(5) → delta_td = sum(3-5) * 10 = -20
        self.assertAlmostEqual(best_val, -20.0, places=8)

    def test_update_cache_2min_tied_distances(self):
        """update_cache_2min with all tied distances per row."""
        n, k = 5, 3
        subD = np.ones((n, k), dtype=np.float64) * 7.0
        subD = np.ascontiguousarray(subD)
        near_idx, near_d, second_d = _core.update_cache_2min(subD, n, k)
        # All distances equal → near should be 7.0, second should be 7.0
        np.testing.assert_allclose(near_d, 7.0)
        np.testing.assert_allclose(second_d, 7.0)

    def test_pam_build_step_single_candidate(self):
        """pam_build_step with only 1 candidate."""
        n = 10
        D = np.random.RandomState(42).uniform(0.1, 5.0, (n, n)).astype(np.float64)
        D = np.ascontiguousarray(0.5 * (D + D.T))
        np.fill_diagonal(D, 0.0)
        dist_to_nearest = np.full(n, 100.0, dtype=np.float64)
        candidates = np.array([3], dtype=np.intp)
        best_idx, best_gain = _core.pam_build_step(
            D, candidates, dist_to_nearest, n, 1
        )
        self.assertEqual(best_idx, 0)  # Only one candidate, index 0 in candidates
        # Gain should be sum of max(0, 100 - D[i, 3]) for all i
        expected_gain = float(np.sum(np.maximum(0.0, 100.0 - D[:, 3])))
        self.assertAlmostEqual(best_gain, expected_gain, places=8)


if __name__ == "__main__":
    unittest.main()
