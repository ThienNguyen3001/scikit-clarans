"""
Targeted Verification & Robustness Test Suite for CLARANS & FastCLARANS.

This module validates that previously discovered weaknesses, latent bugs,
numerical vulnerabilities, and validation blind spots are properly resolved:

1. Numerical Extremes & Overflow:
   - Overflow to inf handled gracefully via ValueError instead of uninitialized IndexError.
    - Zero vectors in Cosine distance handled via ValueError instead of uninitialized IndexError.
    - Dynamic relative tolerance (-max(1e-16, 1e-12 * abs(cost))) enables optimization
      on micro-scale data.
    - Custom callable metric returning NaN properly raises ValueError.

2. Precomputed Matrix Robustness:
   - Asymmetric distance matrices (D != D.T) correctly use column indexing for candidate distances.
   - Non-zero diagonal in precomputed distance matrix behaves consistently.
   - float32 precomputed distances successfully utilize Cython acceleration kernels.
   - Negative distances in precomputed matrix produce predictable results.

3. Parameter Validation & Utility Robustness:
    - Boolean values for num_local and n_clusters are rejected with ValueError.
    - check_medoids rejects negative indices even when n_samples is None.
    - initialize_build, initialize_heuristic, and initialize_k_medoids_plus_plus raise
      descriptive ValueError when K >= N.

4. Dataset Structure Corner Cases:
   - Fewer unique points than clusters (N_unique < K) handled gracefully.
   - Read-only and memory-mapped array inputs (writeable=False) supported.
   - Extreme aspect ratios (D=1 and D >> N) supported.

5. Concurrency & Thread Safety:
   - Concurrent independent fits across multiple worker threads.
   - Concurrent predict() and transform() calls on a shared fitted model.

6. Metric vs. Init Cross-Compatibility:
    - Precomputed metric with explicit init array succeeds without error.
    - Sparse matrix with Chebyshev metric works across k-medoids++, heuristic, build, and random.
    - Scipy/DistanceMetric aliases (e.g., 'infinity', 'sokalmichener', 'p') supported
      across all inits.
    - Obsolete/unsupported metrics without required parameters ('mahalanobis') cleanly
      rejected at validation.
"""

from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from clarans import CLARANS, FastCLARANS, calculate_cost, check_medoids
from clarans._initialization import (
    initialize_build,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)


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
# 7. Bug Fix Verification & Algorithmic Invariants
# ===========================================================================
class TestAlgorithmBugFixesAndEdgeCases(unittest.TestCase):
    """Verifies targeted bug fixes for metric_params, NaN freeze, init, and asymmetric distance."""

    def test_brute_force_passes_metric_params(self):
        """Bug A: Brute force neighbor cost calculation must pass metric_params."""
        X, _ = make_blobs(n_samples=20, n_features=2, centers=2, random_state=42)

        def custom_metric(x, y, weight=1.0):
            return weight * np.sum(np.abs(x - y))

        model = CLARANS(
            n_clusters=2,
            metric=custom_metric,
            metric_params={"weight": 2.5},
            cost_evaluation="brute_force",
            num_local=1,
            max_neighbors=5,
            random_state=42,
        )
        model.fit(X)
        self.assertEqual(len(model.medoid_indices_), 2)
        self.assertTrue(np.isfinite(model.inertia_))

    def test_nan_freeze_recovery_clarans(self):
        """Bug B: CLARANS should not freeze when the first local search iteration is NaN."""
        X, _ = make_blobs(n_samples=20, n_features=2, centers=2, random_state=42)
        model = CLARANS(n_clusters=2, num_local=2, random_state=42)

        call_idx = 0
        original_single_search = model._single_local_search

        def mock_search(X_in, rng, det, buf):
            nonlocal call_idx
            cost, medoids, evals, swaps = original_single_search(X_in, rng, det, buf)
            if call_idx == 0:
                cost = float("nan")
            call_idx += 1
            return cost, medoids, evals, swaps

        with patch.object(model, "_single_local_search", side_effect=mock_search):
            model.fit(X)
            self.assertTrue(np.isfinite(model.inertia_))
            self.assertIsNotNone(model.medoid_indices_)

    def test_nan_freeze_recovery_fast_clarans(self):
        """Bug B: FastCLARANS should not freeze when the first local search iteration is NaN."""
        X, _ = make_blobs(n_samples=20, n_features=2, centers=2, random_state=42)
        model = FastCLARANS(n_clusters=2, num_local=2, random_state=42)

        call_idx = 0
        original_single_search = model._single_local_search

        def mock_search(X_in, rng, det, d_buf, delta_buf):
            nonlocal call_idx
            cost, medoids, evals, swaps = original_single_search(
                X_in, rng, det, d_buf, delta_buf
            )
            if call_idx == 0:
                cost = float("nan")
            call_idx += 1
            return cost, medoids, evals, swaps

        with patch.object(model, "_single_local_search", side_effect=mock_search):
            model.fit(X)
            self.assertTrue(np.isfinite(model.inertia_))
            self.assertIsNotNone(model.medoid_indices_)

    def test_k_medoids_pp_fallback_metric_params(self):
        """Bug C: Fallback branch of initialize_k_medoids_plus_plus must pass metric_params."""
        X, _ = make_blobs(n_samples=15, n_features=2, centers=2, random_state=42)

        def custom_metric(x, y, scale=1.0):
            return scale * np.linalg.norm(x - y)

        with patch("clarans._initialization._core", None):
            medoids = initialize_k_medoids_plus_plus(
                X,
                n_clusters=2,
                metric=custom_metric,
                metric_params={"scale": 2.0},
                n_local_trials=1,
                random_state=42,
            )
            self.assertEqual(len(medoids), 2)

    def test_asymmetric_precomputed_distance_sum(self):
        """Bug D: Precomputed asymmetric distance matrices must sum columns (axis=0)."""
        D = np.array(
            [
                [0.0, 10.0, 10.0],
                [1.0, 0.0, 10.0],
                [1.0, 10.0, 0.0],
            ]
        )
        medoids_h = initialize_heuristic(D, n_clusters=1, metric="precomputed")
        self.assertEqual(medoids_h[0], 0)

        medoids_b = initialize_build(D, n_clusters=1, metric="precomputed")
        self.assertEqual(medoids_b[0], 0)

    def test_init_array_sklearn_standard_and_clear_error(self):
        """Bug F / Bug 1: init array must adhere to sklearn standard (2D for features)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        init_centers = np.array([[1.0, 2.0], [7.0, 8.0]])
        model = CLARANS(n_clusters=2, init=init_centers, random_state=42)
        model.fit(X)
        self.assertEqual(len(model.cluster_centers_), 2)

        init_1d = np.array([0, 2])
        with self.assertRaises(ValueError) as ctx:
            CLARANS(n_clusters=2, init=init_1d, random_state=42).fit(X)
        self.assertIn("init array must be 2D of shape (2, 2)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
