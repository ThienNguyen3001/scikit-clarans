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

from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from clarans import CLARANS, FastCLARANS, calculate_cost, check_medoids, HAS_CYTHON
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
# 1. Numerical Extremes & Overflow Robustness
# ===========================================================================
class TestNumericalVulnerabilitiesAndOverflow(unittest.TestCase):
    """Verifies robustness against numerical extremes, overflow, and tolerance scaling."""

    def test_overflow_data_infinite_cost_best_medoids_uninitialized(self):
        """When data values are extreme (>= 1e160), Euclidean distance
        squared overflows float64 to infinity.

        The model must catch the non-finite cost and raise a clean ValueError
        instead of crashing with an uninitialized IndexError.
        """
        X = np.array([[0.0, 0.0], [1e160, 1e160], [2e160, 2e160]])
        for ModelClass in [CLARANS, FastCLARANS]:
            with self.assertRaises(ValueError) as ctx:
                ModelClass(n_clusters=2, random_state=42).fit(X)
            self.assertIn("non-finite cost", str(ctx.exception).lower())

    def test_cosine_metric_zero_vector_nan_cost(self):
        """Cosine distance involving a zero-vector produces NaN.

        The model must catch the non-finite cost and raise a clean ValueError
        instead of crashing with an uninitialized IndexError.
        """
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        for ModelClass in [CLARANS, FastCLARANS]:
            with self.assertRaises(ValueError) as ctx:
                ModelClass(n_clusters=2, metric="cosine", random_state=42).fit(X)
            self.assertIn("non-finite cost", str(ctx.exception).lower())

    def test_micro_scale_data_delta_tolerance_freeze(self):
        """Dynamic relative tolerance allows successful medoid swaps on micro-scale
        data where costs and deltas are in the order of 1e-13 to 1e-14.
        """
        rng = np.random.RandomState(42)
        X_normal = rng.randn(50, 2)

        # Normal scale yields successful swaps
        m_normal = CLARANS(
            n_clusters=3, init="random", max_neighbors=200, random_state=42
        ).fit(X_normal)
        self.assertGreater(
            m_normal.n_swaps_, 0, "Normal scale should perform swaps"
        )

        # Micro scale now also succeeds thanks to dynamic tolerance
        X_micro = X_normal * 1e-13
        m_micro = CLARANS(
            n_clusters=3, init="random", max_neighbors=200, random_state=42
        ).fit(X_micro)
        self.assertGreater(
            m_micro.n_swaps_,
            0,
            "Micro-scale data should perform swaps with dynamic tolerance",
        )

        fm_micro = FastCLARANS(
            n_clusters=3, init="random", max_neighbors=200, random_state=42
        ).fit(X_micro)
        self.assertGreater(
            fm_micro.n_swaps_,
            0,
            "FastCLARANS micro-scale data should perform swaps with dynamic tolerance",
        )

    def test_custom_metric_returning_nan_behavior(self):
        """A custom callable distance metric returning NaN is safely caught
        and raises a descriptive ValueError.
        """
        def nan_metric(u, v):
            return np.nan

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        for ModelClass in [CLARANS, FastCLARANS]:
            with self.assertRaises(ValueError) as ctx:
                ModelClass(n_clusters=2, metric=nan_metric, random_state=42).fit(X)
            self.assertIn("non-finite cost", str(ctx.exception).lower())


# ===========================================================================
# 2. Precomputed Matrix Robustness
# ===========================================================================
class TestPrecomputedMatrixFlaws(unittest.TestCase):
    """Verifies correctness and efficiency for precomputed distance matrices."""

    def test_asymmetric_precomputed_matrix_discrepancy(self):
        """Asymmetric distance matrices (D != D.T) must use column extraction
        D[:, candidate] so that medoids_dist matches the definition of distance
        from all samples to the medoid.

        Model inertia_ must match calculate_cost(D, medoids) within numerical tolerance.
        """
        rng = np.random.RandomState(42)
        D = rng.rand(20, 20)
        np.fill_diagonal(D, 0.0)

        for ModelClass in [CLARANS, FastCLARANS]:
            m = ModelClass(
                n_clusters=4,
                metric="precomputed",
                num_local=5,
                max_neighbors=100,
                random_state=42,
            ).fit(D)

            true_cost = calculate_cost(D, m.medoid_indices_, metric="precomputed")
            discrepancy = abs(m.inertia_ - true_cost)
            self.assertLess(
                discrepancy,
                1e-5,
                f"{ModelClass.__name__} inertia_ should match calculate_cost on "
                "asymmetric matrices",
            )

    def test_precomputed_partially_asymmetric_matrix_detected(self):
        """A matrix that is symmetric everywhere except at index >= 100 must be
        deterministically identified as asymmetric by the pure C kernel.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(120, 4)
        D_sym = pairwise_distances(X, metric="euclidean")

        # Perfectly symmetric matrix should be marked symmetric
        m_sym = CLARANS(n_clusters=3, metric="precomputed", random_state=42).fit(D_sym)
        self.assertTrue(m_sym._precomputed_is_sym)

        # Perturb only point 101 (pairs in 0..99 remain symmetric)
        D_asym = D_sym.copy()
        D_asym[101, 5] += 10.0  # D[101, 5] != D[5, 101]

        for ModelClass in [CLARANS, FastCLARANS]:
            m = ModelClass(
                n_clusters=3,
                metric="precomputed",
                num_local=2,
                max_neighbors=50,
                random_state=42,
            ).fit(D_asym)

            self.assertFalse(
                m._precomputed_is_sym,
                f"{ModelClass.__name__} failed to detect asymmetry at index 101",
            )
            true_cost = calculate_cost(D_asym, m.medoid_indices_, metric="precomputed")
            self.assertLess(
                abs(m.inertia_ - true_cost),
                1e-5,
                f"{ModelClass.__name__} cost discrepancy on asymmetric matrix",
            )

    def test_precomputed_nonzero_diagonal_self_cluster_inversion(self):
        """If precomputed distance matrix has D_ii > D_ij, medoids
        are assigned to other medoids' clusters instead of their own cluster.
        """
        D = np.ones((5, 5)) * 2.0
        np.fill_diagonal(D, 5.0)

        m = CLARANS(n_clusters=2, metric="precomputed", random_state=42).fit(D)
        for i, med_idx in enumerate(m.medoid_indices_):
            self.assertNotEqual(
                m.labels_[med_idx],
                i,
                "Medoid was unexpectedly assigned to its own cluster despite D_ii > D_ij",
            )

    def test_float32_precomputed_cython_bypass_flaw(self):
        """float32 precomputed distance matrix must utilize Cython acceleration
        by matching buffer dtype to input dtype.
        """
        try:
            from clarans import _core
            if _core is None:
                self.skipTest("Cython _core extension is not compiled.")
        except ImportError:
            self.skipTest("Cython _core extension is not available.")

        X = np.random.RandomState(42).randn(30, 4)
        D_f32 = pairwise_distances(X).astype(np.float32)

        with patch.object(_core, "clarans_delta", wraps=_core.clarans_delta) as mock_clarans:
            CLARANS(n_clusters=3, metric="precomputed", random_state=42).fit(D_f32)
            self.assertGreater(
                mock_clarans.call_count,
                0,
                "clarans_delta should be called for float32 precomputed matrices",
            )

        with patch.object(_core, "fastpam1_delta", wraps=_core.fastpam1_delta) as mock_fastpam:
            FastCLARANS(n_clusters=3, metric="precomputed", random_state=42).fit(D_f32)
            self.assertGreater(
                mock_fastpam.call_count,
                0,
                "fastpam1_delta should be called for float32 precomputed matrices",
            )

    def test_negative_distances_in_precomputed_matrix(self):
        """Precomputed matrices with negative entries produce predictable negative inertia."""
        D = np.array([
            [0.0, -1.0, -2.0],
            [-1.0, 0.0, -3.0],
            [-2.0, -3.0, 0.0],
        ])
        m = CLARANS(n_clusters=2, metric="precomputed", random_state=42).fit(D)
        self.assertLess(
            m.inertia_, 0.0, "Model accepted negative distances and produced negative inertia"
        )


# ===========================================================================
# 3. Parameter Validation & Utility Robustness
# ===========================================================================
class TestTypeValidationAndUtilityBlindSpots(unittest.TestCase):
    """Verifies parameter validation rigor and standalone utility correctness."""

    def test_bool_num_local_validation_bypass(self):
        """num_local and n_clusters must reject boolean values (True, False)."""
        X = np.random.RandomState(42).randn(15, 2)

        for bool_val in [True, False]:
            with self.assertRaises(ValueError):
                CLARANS(num_local=bool_val, random_state=42).fit(X)
            with self.assertRaises(ValueError):
                FastCLARANS(num_local=bool_val, random_state=42).fit(X)
            with self.assertRaises(ValueError):
                CLARANS(n_clusters=bool_val, random_state=42).fit(X)
            with self.assertRaises(ValueError):
                FastCLARANS(n_clusters=bool_val, random_state=42).fit(X)

    def test_check_medoids_negative_indices_unbounded(self):
        """check_medoids must reject negative indices even when n_samples is None."""
        medoids_with_neg = [-1, 0, 2]
        with self.assertRaises(ValueError) as ctx:
            check_medoids(medoids_with_neg)
        self.assertIn("negative", str(ctx.exception).lower())

        with self.assertRaises(ValueError):
            check_medoids(medoids_with_neg, n_samples=10)

    def test_initialize_build_k_greater_n_raises_index_error(self):
        """initialize_build must raise a descriptive ValueError when n_clusters >= n_samples."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError) as ctx:
            initialize_build(X, n_clusters=3)
        self.assertIn("n_clusters", str(ctx.exception).lower())

    def test_initialize_heuristic_and_kmedoids_pp_boundary(self):
        """initialize_heuristic and initialize_k_medoids_plus_plus raise descriptive
        ValueError when K >= N.
        """
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            initialize_heuristic(X, n_clusters=3)

        with self.assertRaises(ValueError):
            initialize_k_medoids_plus_plus(X, n_clusters=3)


# ===========================================================================
# 4. Dataset Structure Corner Cases
# ===========================================================================
class TestDataStructureCornerCases(unittest.TestCase):
    """Exposes edge cases relating to data layout, duplicates, and dimensionality."""

    def test_fewer_unique_points_than_clusters_empty_clusters(self):
        """A dataset with N=4 points containing only 2 distinct coordinates,
        clustered into K=3 medoids.

        At least two medoids will have identical coordinates. Empty clusters
        must be handled gracefully by predict() and transform().
        """
        X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
        for ModelClass in [CLARANS, FastCLARANS]:
            m = ModelClass(n_clusters=3, random_state=42).fit(X)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertAlmostEqual(m.inertia_, 0.0)

            unique_labels = np.unique(m.labels_)
            self.assertLess(
                len(unique_labels),
                3,
                f"{ModelClass.__name__}: Expected an empty cluster due to duplicate points",
            )

            preds = m.predict(X)
            np.testing.assert_array_equal(preds, m.labels_)
            trans = m.transform(X)
            self.assertEqual(trans.shape, (4, 3))
            self.assertTrue(np.all(np.isfinite(trans)))

    def test_readonly_and_memmap_input_arrays(self):
        """Verify that CLARANS and FastCLARANS do NOT mutate the input array in-place,
        supporting read-only (memmap) buffers.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(30, 4)
        X.flags.writeable = False

        for ModelClass in [CLARANS, FastCLARANS]:
            m = ModelClass(n_clusters=3, random_state=42).fit(X)
            self.assertEqual(len(m.medoid_indices_), 3)
            preds = m.predict(X)
            self.assertEqual(preds.shape, (30,))
            trans = m.transform(X)
            self.assertEqual(trans.shape, (30, 3))

    def test_single_feature_data_extremes(self):
        """Single feature data (D=1) and extreme high dimensional data (D >> N)."""
        rng = np.random.RandomState(42)

        # 1D feature data
        X_1d = rng.randn(25, 1)
        for ModelClass in [CLARANS, FastCLARANS]:
            m = ModelClass(n_clusters=2, random_state=42).fit(X_1d)
            self.assertEqual(m.cluster_centers_.shape, (2, 1))

        # High dimensional data (D >> N)
        X_hd = rng.randn(10, 500)
        for ModelClass in [CLARANS, FastCLARANS]:
            m = ModelClass(n_clusters=2, random_state=42).fit(X_hd)
            self.assertEqual(m.cluster_centers_.shape, (2, 500))


# ===========================================================================
# 5. Concurrency & Thread Safety
# ===========================================================================
class TestConcurrencyAndThreadSafety(unittest.TestCase):
    """Verifies behavior and thread safety under concurrent execution."""

    def test_concurrent_independent_fits_multithreading(self):
        """Running multiple independent CLARANS / FastCLARANS instances in parallel
        across threads must produce valid, consistent results.
        """
        X, _ = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42)

        def worker(seed):
            m = FastCLARANS(n_clusters=3, random_state=seed, num_local=2)
            m.fit(X)
            return m.inertia_, len(m.medoid_indices_)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(12)))

        self.assertEqual(len(results), 12)
        for inertia, n_medoids in results:
            self.assertEqual(n_medoids, 3)
            self.assertTrue(np.isfinite(inertia))
            self.assertGreater(inertia, 0.0)

    def test_concurrent_predict_transform_shared_fitted_model(self):
        """Calling predict() and transform() simultaneously on a single shared fitted
        model instance from multiple threads must be thread-safe.
        """
        X_train, _ = make_blobs(n_samples=80, n_features=3, centers=3, random_state=42)
        model = FastCLARANS(n_clusters=3, random_state=42).fit(X_train)

        def inference_worker(seed):
            rng = np.random.RandomState(seed)
            X_test = rng.randn(20, 3)
            preds = model.predict(X_test)
            trans = model.transform(X_test)
            return preds.shape, trans.shape

        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(inference_worker, range(24)))

        self.assertEqual(len(results), 24)
        for p_shape, t_shape in results:
            self.assertEqual(p_shape, (20,))
            self.assertEqual(t_shape, (20, 3))


# ===========================================================================
# 6. Metric vs. Init Cross-Compatibility
# ===========================================================================
class TestMetricInitCrossCompatibility(unittest.TestCase):
    """Verifies compatibility between supported metrics and initialization strategies."""

    def test_precomputed_metric_with_explicit_array_init_crashes(self):
        """Passing metric='precomputed' alongside an explicit init array (either 1D indices
        or 2D distance submatrices) must succeed cleanly.
        """
        X = np.random.RandomState(42).randn(10, 2)
        D = pairwise_distances(X)

        # 2D submatrix init
        init_arr_2d = D[:2].copy()
        for ModelClass in [CLARANS, FastCLARANS]:
            m = ModelClass(
                n_clusters=2, metric="precomputed", init=init_arr_2d, random_state=42
            ).fit(D)
            self.assertEqual(len(m.medoid_indices_), 2)

        # 1D index array init
        init_arr_1d = np.array([0, 3])
        for ModelClass in [CLARANS, FastCLARANS]:
            m = ModelClass(
                n_clusters=2, metric="precomputed", init=init_arr_1d, random_state=42
            ).fit(D)
            self.assertEqual(len(m.medoid_indices_), 2)

    def test_sparse_matrix_chebyshev_init_inconsistency(self):
        """On sparse matrices (CSR), metric='chebyshev' must succeed across
        all initialization strategies: 'random', 'k-medoids++', 'heuristic', and 'build'.
        """
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("scipy is not installed")

        X_csr = sparse.csr_matrix(np.random.RandomState(42).randn(8, 3))

        for init_strategy in ["random", "k-medoids++", "heuristic", "build"]:
            m = CLARANS(
                n_clusters=2,
                metric="chebyshev",
                init=init_strategy,
                random_state=42,
            ).fit(X_csr)
            self.assertEqual(len(m.medoid_indices_), 2)

    def test_dm_metrics_pairwise_distances_delegation_incompatibility(self):
        """Metrics from DistanceMetric (e.g. 'infinity', 'sokalmichener', 'p')
        must succeed across distance-based initializations.
        """
        X = np.random.RandomState(42).randn(12, 3)

        for init_strategy in ["random", "k-medoids++", "heuristic", "build"]:
            m = CLARANS(
                n_clusters=2,
                metric="infinity",
                init=init_strategy,
                random_state=42,
            ).fit(X)
            self.assertEqual(len(m.medoid_indices_), 2)

    def test_mahalanobis_unsupported_due_to_missing_vi_parameter(self):
        """Unsupported metrics requiring extra parameters ('mahalanobis')
        must be cleanly rejected by parameter validation with a descriptive message.
        """
        X = np.random.RandomState(42).randn(10, 3)
        for ModelClass in [CLARANS, FastCLARANS]:
            with self.assertRaises(ValueError) as ctx:
                ModelClass(n_clusters=2, metric="mahalanobis", random_state=42).fit(X)
            self.assertIn("metric", str(ctx.exception).lower())


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
    (get_params, set_params, repr, tags, n_features_in_)."""

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
        expected = max(1, int(250 / 3), int(0.025 * (100 - 3)))
        self.assertEqual(m.max_neighbors_, expected)

        # The values differ between algorithms across sample sizes
        clarans_expected = max(250, int(0.0125 * 10 * (50000 - 10)))
        X_large, _ = make_blobs(n_samples=50000, centers=10, random_state=42)
        m_fast = FastCLARANS(n_clusters=10, random_state=42)
        m_fast.fit(X_large)
        fast_expected = max(1, int(250 / 10), int(0.025 * (50000 - 10)))
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
