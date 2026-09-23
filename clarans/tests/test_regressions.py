"""Comprehensive Regression Test Suite for scikit-clarans.

Consolidates all regression tests for reported bugs, audit findings, and edge-case
fixes to permanently prevent regressions across releases.

Sections:
1. Historical Bug Fixes (Bugs #4, #6, #10, #11, #13, and Core Regressions)
2. Audit Bug Fixes (Asymmetric k-medoids++, Array-like inits, NaN tolerance, Score API)
3. Deep Audit Bug Fixes (Callable metric direction, IEEE-754 signed Inf, NaN recovery)
4. Algorithmic Edge-Case Regressions (Brute force params, NaN freeze recovery, Asymmetric sum)
"""

import threading
import unittest
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV

from clarans import CLARANS, FastCLARANS, calculate_cost
from clarans._initialization import (
    initialize_build,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)
from clarans.utils import HAS_CYTHON

if HAS_CYTHON:
    from clarans._core import (
        fastpam1_delta,
        is_matrix_symmetric,
        update_cache_2min,
    )


# ============================================================================
# Section 1: Historical Bug Fixes (Bugs #4, #6, #10, #11, #13, Regressions)
# ============================================================================
class TestBug4Fix_MemoryLeakCleanup(unittest.TestCase):
    """Verify that _precomputed_source is properly cleaned up when fit() fails."""

    def test_clarans_precomputed_cleanup_on_failure(self):
        """CLARANS: _precomputed_source must NOT leak after fit failure."""
        rng = np.random.RandomState(42)
        D = pairwise_distances(rng.randn(20, 3))
        model = CLARANS(n_clusters=3, metric="precomputed", random_state=42)

        def mock_search(self_inner, X_in, rng_in, det, buf):
            return float("inf"), np.array([0, 1, 2]), 10, 0

        with patch.object(CLARANS, "_single_local_search", mock_search):
            with self.assertRaises(ValueError):
                model.fit(D)

        self.assertFalse(
            hasattr(model, "_precomputed_source"),
            "_precomputed_source should be cleaned up after fit failure",
        )

    def test_fastclarans_precomputed_cleanup_on_failure(self):
        """FastCLARANS: _precomputed_source must NOT leak after fit failure."""
        rng = np.random.RandomState(42)
        D = pairwise_distances(rng.randn(20, 3))
        model = FastCLARANS(n_clusters=3, metric="precomputed", random_state=42)

        def mock_search(self_inner, X_in, rng_in, det, buf, delta_buf):
            return float("inf"), np.array([0, 1, 2]), 10, 0

        with patch.object(FastCLARANS, "_single_local_search", mock_search):
            with self.assertRaises(ValueError):
                model.fit(D)

        self.assertFalse(
            hasattr(model, "_precomputed_source"),
            "_precomputed_source should be cleaned up after fit failure",
        )

    def test_successful_fit_precomputed_source_is_cleaned_up(self):
        """After successful fit, _precomputed_source must be cleaned up."""
        rng = np.random.RandomState(42)
        D = pairwise_distances(rng.randn(20, 3))
        for cls in [CLARANS, FastCLARANS]:
            model = cls(n_clusters=3, metric="precomputed", random_state=42, num_local=1)
            model.fit(D)
            self.assertFalse(
                hasattr(model, "_precomputed_source"),
                f"{cls.__name__}: _precomputed_source must be deleted after fit",
            )


class TestBug6Fix_HeuristicDeterministicOrder(unittest.TestCase):
    """Verify that initialize_heuristic returns deterministically sorted medoids."""

    def test_heuristic_output_is_sorted(self):
        """Medoid indices from initialize_heuristic should always be sorted."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)

        for k in [3, 5, 10]:
            medoids = initialize_heuristic(X, n_clusters=k, metric="euclidean")
            np.testing.assert_array_equal(
                medoids, np.sort(medoids),
                err_msg=f"k={k}: medoid indices should be sorted",
            )

    def test_heuristic_reproducible_across_calls(self):
        """Same data should always produce identical medoid arrays."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)

        results = [
            tuple(initialize_heuristic(X, n_clusters=8, metric="euclidean"))
            for _ in range(20)
        ]

        self.assertEqual(
            len(set(results)), 1,
            "All calls should return identical medoid arrays",
        )


class TestBug10Fix_DeadCodeRemoved(unittest.TestCase):
    """Verify that _DELTA_TOL dead code has been removed."""

    def test_delta_tol_no_longer_defined(self):
        """_DELTA_TOL should not exist in the module."""
        from clarans import _clarans

        self.assertFalse(
            hasattr(_clarans, "_DELTA_TOL"),
            "_DELTA_TOL dead code should be removed",
        )


class TestBug11Fix_ArgpartitionPerformance(unittest.TestCase):
    """Verify that _compute_2min Python fallback uses argpartition."""

    def test_compute_2min_uses_argpartition(self):
        """Fallback path should use O(n*k) argpartition, not O(n*k*log k) argsort."""
        import inspect
        source = inspect.getsource(CLARANS._compute_2min)

        self.assertIn("argpartition", source, "Should use argpartition")
        self.assertNotIn("argsort", source, "Should NOT use argsort")

    def test_compute_2min_correctness_after_fix(self):
        """Verify argpartition gives correct 2-min results."""
        X, _ = make_blobs(n_samples=50, centers=5, random_state=42)

        model = CLARANS(n_clusters=5, random_state=42)
        model.fit(X)

        medoids_dist = model._compute_medoids_distances(X, model.medoid_indices_)

        with patch("clarans._clarans._core", None):
            near_idx, near_dist, second_dist = model._compute_2min(medoids_dist)

        for i in range(len(X)):
            dists = medoids_dist[i]
            sorted_dists = np.sort(dists)
            sorted_indices = np.argsort(dists)

            self.assertEqual(near_idx[i], sorted_indices[0],
                             f"Sample {i}: wrong nearest medoid index")
            self.assertAlmostEqual(near_dist[i], sorted_dists[0],
                                   msg=f"Sample {i}: wrong nearest distance")
            self.assertAlmostEqual(second_dist[i], sorted_dists[1],
                                   msg=f"Sample {i}: wrong second distance")


class TestBug13Fix_ThreadSafeWarning(unittest.TestCase):
    """Verify that warning mechanism uses double-checked locking."""

    def test_warning_uses_lock(self):
        """Verify threading.Lock is used for warning flag."""
        from clarans import utils

        self.assertTrue(
            hasattr(utils, "_cython_warning_lock"),
            "Should have a threading.Lock for warning synchronization",
        )
        self.assertIsInstance(
            utils._cython_warning_lock,
            type(threading.Lock()),
            "_cython_warning_lock should be a Lock instance",
        )

    def test_double_checked_locking_pattern(self):
        """Source should use double-checked locking pattern."""
        import inspect
        from clarans.utils import _warn_cython_unavailable

        source = inspect.getsource(_warn_cython_unavailable)

        self.assertIn("_cython_warning_lock", source, "Should reference the lock")
        flag_checks = source.count("_cython_warning_issued")
        self.assertGreaterEqual(
            flag_checks, 2,
            "Should check _cython_warning_issued at least twice (double-checked locking)",
        )


class TestRegressions(unittest.TestCase):
    """Ensure bug fixes don't break normal clustering behavior."""

    def test_clarans_basic_fit_predict(self):
        """Basic fit-predict workflow still works."""
        X, _ = make_blobs(n_samples=80, centers=3, random_state=42)
        model = CLARANS(n_clusters=3, random_state=42)
        model.fit(X)

        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertTrue(np.isfinite(model.inertia_))

        labels = model.predict(X)
        self.assertEqual(labels.shape, (80,))

        transformed = model.transform(X)
        self.assertEqual(transformed.shape, (80, 3))

    def test_fastclarans_basic_fit_predict(self):
        """Basic FastCLARANS workflow still works."""
        X, _ = make_blobs(n_samples=80, centers=3, random_state=42)
        model = FastCLARANS(n_clusters=3, random_state=42)
        model.fit(X)

        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertTrue(np.isfinite(model.inertia_))

    def test_precomputed_distance_still_works(self):
        """Precomputed distance matrix workflow still works."""
        X, _ = make_blobs(n_samples=40, centers=3, random_state=42)
        D = pairwise_distances(X)

        for ModelClass in [CLARANS, FastCLARANS]:
            model = ModelClass(n_clusters=3, metric="precomputed", random_state=42)
            model.fit(D)

            self.assertEqual(len(model.medoid_indices_), 3)
            self.assertTrue(np.isfinite(model.inertia_))

            true_cost = calculate_cost(D, model.medoid_indices_, metric="precomputed")
            self.assertAlmostEqual(model.inertia_, true_cost, places=5)

    def test_heuristic_init_still_works(self):
        """Heuristic initialization still produces valid results."""
        X, _ = make_blobs(n_samples=60, centers=3, random_state=42)

        model = CLARANS(n_clusters=3, init="heuristic", num_local=1, random_state=42)
        model.fit(X)

        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertTrue(np.isfinite(model.inertia_))

    def test_all_init_methods_work(self):
        """All initialization methods produce valid results after fix."""
        X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

        for init in ["random", "k-medoids++", "heuristic", "build"]:
            for ModelClass in [CLARANS, FastCLARANS]:
                model = ModelClass(
                    n_clusters=3, init=init, num_local=1, max_neighbors=50, random_state=42
                )
                model.fit(X)
                self.assertEqual(
                    len(model.medoid_indices_), 3,
                    f"{ModelClass.__name__} init={init} failed",
                )


# ============================================================================
# Section 2: Audit Bug Fixes
# ============================================================================
class TestAuditBugfixes:
    def test_kmedoids_pp_asymmetric_precomputed_direction(self):
        """Verify k-medoids++ uses column indexing (sample -> medoid) on asymmetric matrix."""
        D = np.array([
            [0.0, 0.1, 100.0, 100.0],
            [100.0, 0.0, 50.0, 50.0],
            [0.1, 50.0, 0.0, 50.0],
            [0.1, 50.0, 50.0, 0.0],
        ], dtype=np.float64)

        selected_candidates = []
        for seed in range(50):
            r = np.random.RandomState(seed)
            medoids = initialize_k_medoids_plus_plus(
                D, n_clusters=2, random_state=r, metric="precomputed"
            )
            if medoids[0] == 0:
                selected_candidates.append(medoids[1])

        count_1 = selected_candidates.count(1)
        count_2 = selected_candidates.count(2)
        assert count_1 > count_2

    def test_init_accepts_tuple_and_range(self):
        """Verify init accepts 2D tuple for feature space, and 1D tuple/range for precomputed."""
        X, _ = make_blobs(n_samples=30, centers=2, n_features=2, random_state=42)

        init_tuples = ((float(X[0, 0]), float(X[0, 1])), (float(X[1, 0]), float(X[1, 1])))
        m1 = CLARANS(n_clusters=2, init=init_tuples, num_local=1, random_state=42).fit(X)
        assert len(m1.medoid_indices_) == 2
        m1_fast = FastCLARANS(
            n_clusters=2, init=init_tuples, num_local=1, random_state=42
        ).fit(X)
        assert len(m1_fast.medoid_indices_) == 2

        D = pairwise_distances(X)
        m2 = CLARANS(
            n_clusters=2, init=(0, 1), metric="precomputed", num_local=1, random_state=42
        ).fit(D)
        assert len(m2.medoid_indices_) == 2
        m2_fast = FastCLARANS(
            n_clusters=2, init=(0, 1), metric="precomputed", num_local=1, random_state=42
        ).fit(D)
        assert len(m2_fast.medoid_indices_) == 2

        m3 = CLARANS(
            n_clusters=2, init=range(2), metric="precomputed", num_local=1, random_state=42
        ).fit(D)
        assert len(m3.medoid_indices_) == 2
        m3_fast = FastCLARANS(
            n_clusters=2, init=range(2), metric="precomputed", num_local=1, random_state=42
        ).fit(D)
        assert len(m3_fast.medoid_indices_) == 2

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython core not available")
    def test_is_matrix_symmetric_nan_and_scale(self):
        """Verify is_matrix_symmetric handles NaNs correctly and supports rtol/atol."""
        nan_matrix = np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float64)
        assert not is_matrix_symmetric(nan_matrix, 2)

        nan_asym = np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float64)
        assert not is_matrix_symmetric(nan_asym, 2)

        large_val = 1e6
        D = np.full((5, 5), large_val, dtype=np.float64)
        np.fill_diagonal(D, 0.0)
        D[0, 1] += 2.0
        assert is_matrix_symmetric(D, 5, rtol=1e-5, atol=1e-8)

        D[0, 1] += 18.0
        assert not is_matrix_symmetric(D, 5, rtol=1e-5, atol=1e-8)

    def test_score_method_and_gridsearch_default(self):
        """Verify score(X) returns negative inertia and works seamlessly in GridSearchCV."""
        X, _ = make_blobs(n_samples=60, centers=3, n_features=2, random_state=42)
        model = CLARANS(n_clusters=3, random_state=42)

        with pytest.raises(NotFittedError):
            model.score(X)

        model.fit(X)
        assert np.isclose(model.score(X), -model.inertia_)

        X_test, _ = make_blobs(n_samples=20, centers=3, n_features=2, random_state=99)
        test_score = model.score(X_test)
        assert isinstance(test_score, float)
        assert test_score < 0

        fast_model = FastCLARANS(n_clusters=3, random_state=42).fit(X)
        assert np.isclose(fast_model.score(X), -fast_model.inertia_)

        grid = GridSearchCV(
            CLARANS(random_state=42, num_local=1),
            param_grid={"n_clusters": [2, 3]},
            cv=2,
        )
        grid.fit(X)
        assert grid.best_params_["n_clusters"] in [2, 3]

    def test_nan_euclidean_end_to_end(self):
        """Verify metric='nan_euclidean' works on missing data without ValueError."""
        rng = np.random.RandomState(42)
        X, _ = make_blobs(n_samples=40, centers=2, n_features=3, random_state=42)

        mask = rng.rand(*X.shape) < 0.05
        X[mask] = np.nan

        model = CLARANS(n_clusters=2, metric="nan_euclidean", random_state=42)
        model.fit(X)

        assert len(model.medoid_indices_) == 2
        labels = model.predict(X)
        assert labels.shape == (40,)

        dists = model.transform(X)
        assert dists.shape == (40, 2)
        assert not np.isnan(dists).any()

        score = model.score(X)
        assert np.isfinite(score)

        tags = model.__sklearn_tags__()
        assert tags.input_tags.allow_nan is True

    def test_precomputed_fortran_keeps_cython_active(self):
        """Verify precomputed F-contiguous distance matrix is handled cleanly."""
        D = np.array([
            [0.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
            [3.0, 1.0, 0.0],
        ], dtype=np.float64, order="F")

        model = CLARANS(n_clusters=2, metric="precomputed", random_state=42)
        model.fit(D)
        assert len(model.medoid_indices_) == 2
        assert model.labels_.shape == (3,)

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython core not available")
    def test_fastpam1_delta_buffer_overflow_check(self):
        """Verify fastpam1_delta checks delta_buf length against n_clusters."""
        n_clusters = 5
        n_samples = 20
        delta_buf = np.zeros(3, dtype=np.float64)  # too small: 3 < 5
        d_xc = np.zeros(n_samples, dtype=np.float64)
        near_idx_map = np.zeros(n_samples, dtype=np.intp)
        near_dist = np.zeros(n_samples, dtype=np.float64)
        second_dist = np.zeros(n_samples, dtype=np.float64)

        with pytest.raises(ValueError, match="delta_buf length .* must be >= n_clusters"):
            fastpam1_delta(
                near_idx_map,
                near_dist,
                second_dist,
                d_xc,
                n_samples,
                n_clusters,
                delta_buf=delta_buf,
            )


# ============================================================================
# Section 3: Deep Audit Bug Fixes
# ============================================================================
class TestDeepAuditBugs:
    def test_asymmetric_callable_metric_inertia_consistency(self):
        """CLARANS and FastCLARANS must compute candidate distances as d(sample -> candidate)."""
        def asymmetric_metric(x, y):
            diff = x[0] - y[0]
            return diff * 10.0 if diff > 0 else -diff * 1.0

        rng = np.random.RandomState(42)
        X = rng.uniform(0, 100, (30, 1))

        for ModelClass in [CLARANS, FastCLARANS]:
            model = ModelClass(
                n_clusters=3,
                metric=asymmetric_metric,
                max_neighbors=50,
                num_local=2,
                random_state=42,
            )
            model.fit(X)

            dists = model.transform(X)
            expected_inertia = float(np.sum(np.min(dists, axis=1)))
            np.testing.assert_allclose(
                model.inertia_,
                expected_inertia,
                rtol=1e-7,
                atol=1e-7,
                err_msg=f"{ModelClass.__name__} inertia_ must match actual clustering cost",
            )
            assert np.isclose(model.score(X), -expected_inertia)

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython core not available")
    def test_is_matrix_symmetric_infinities(self):
        """Cython is_matrix_symmetric must properly distinguish +inf from -inf."""
        D_matching_inf = np.array([
            [0.0, np.inf, 2.0],
            [np.inf, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ], dtype=np.float64)
        assert is_matrix_symmetric(D_matching_inf, 3)

        D_mixed_inf = np.array([
            [0.0, np.inf, 2.0],
            [-np.inf, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ], dtype=np.float64)
        assert not is_matrix_symmetric(D_mixed_inf, 3)

        D_inf_vs_finite = np.array([
            [0.0, np.inf, 2.0],
            [1e308, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ], dtype=np.float64)
        assert not is_matrix_symmetric(D_inf_vs_finite, 3)

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython core not available")
    def test_update_cache_2min_nan_recovery(self):
        """Cython update_cache_2min second-nearest distance must not remain NaN."""
        subD = np.array([
            [np.nan, 2.0, 5.0],
            [3.0, np.nan, 1.0],
            [4.0, 6.0, np.nan],
        ], dtype=np.float64)

        near_idx, near_dist, second_dist = update_cache_2min(subD, 3, 3)

        assert not np.isnan(near_dist[0])
        assert not np.isnan(second_dist[0])
        assert near_dist[0] == 2.0
        assert second_dist[0] == 5.0
        assert near_idx[0] == 1

        assert near_dist[1] == 1.0
        assert second_dist[1] == 3.0
        assert near_idx[1] == 2

    def test_calculate_cost_validation(self):
        """calculate_cost must validate medoid_indices using check_medoids."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        with pytest.raises(ValueError, match="empty"):
            calculate_cost(X, [])

        with pytest.raises(ValueError, match="non-negative"):
            calculate_cost(X, [-1])

        with pytest.raises(ValueError, match="within"):
            calculate_cost(X, [9999])

        with pytest.raises(ValueError, match="duplicate"):
            calculate_cost(X, [0, 0])


# ============================================================================
# Section 4: Algorithmic Edge-Case Regressions
# ============================================================================
class TestAlgorithmBugFixesAndEdgeCases(unittest.TestCase):
    """Regressions for algorithm fixes: NaN freezes, fallback metric params, and asymmetric sums."""

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
        """Bug C: FastCLARANS should not freeze when the first local search iteration is NaN."""
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
        """Bug D: Fallback branch of initialize_k_medoids_plus_plus must pass metric_params."""
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
        """Bug E: Precomputed asymmetric distance matrices must sum columns (axis=0)."""
        D = np.array([
            [0.0, 10.0, 10.0],
            [1.0, 0.0, 10.0],
            [1.0, 10.0, 0.0],
        ])
        medoids_h = initialize_heuristic(D, n_clusters=1, metric="precomputed")
        self.assertEqual(medoids_h[0], 0)

        medoids_b = initialize_build(D, n_clusters=1, metric="precomputed")
        self.assertEqual(medoids_b[0], 0)

    def test_init_array_sklearn_standard_and_clear_error(self):
        """Bug F: init array must adhere to sklearn standard (2D for features)."""
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
    unittest.main(verbosity=2)
