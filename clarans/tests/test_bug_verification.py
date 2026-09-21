"""
Bug Verification Test Suite — v3 (post-fix verification).

Tests are now written to verify the bugs are FIXED:
- PASS = Bug is fixed correctly
- FAIL = Bug still present or fix broke something
"""

import threading
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from clarans import CLARANS, FastCLARANS, calculate_cost
from clarans._initialization import (
    initialize_heuristic,
)

try:
    from clarans import _core
except ImportError:
    _core = None


# ============================================================================
# Bug #4 Fix: _precomputed_source cleaned up on fit failure
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
            "_precomputed_source should be cleaned up after fit failure"
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
            "_precomputed_source should be cleaned up after fit failure"
        )

    def test_successful_fit_still_cleans_precomputed_source(self):
        """Normal successful fit should still clean up _precomputed_source."""
        rng = np.random.RandomState(42)
        D = pairwise_distances(rng.randn(20, 3))

        for ModelClass in [CLARANS, FastCLARANS]:
            model = ModelClass(n_clusters=3, metric="precomputed", random_state=42)
            model.fit(D)

            self.assertFalse(
                hasattr(model, "_precomputed_source"),
                f"{ModelClass.__name__}: _precomputed_source should be deleted after successful fit"
            )
            # But model should still be functional
            self.assertTrue(np.isfinite(model.inertia_))
            self.assertEqual(len(model.medoid_indices_), 3)


# ============================================================================
# Bug #6 Fix: initialize_heuristic returns sorted output
# ============================================================================
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
                err_msg=f"k={k}: medoid indices should be sorted"
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
            "All calls should return identical medoid arrays"
        )


# ============================================================================
# Bug #10 Fix: _DELTA_TOL removed
# ============================================================================
class TestBug10Fix_DeadCodeRemoved(unittest.TestCase):
    """Verify that _DELTA_TOL dead code has been removed."""

    def test_delta_tol_no_longer_defined(self):
        """_DELTA_TOL should not exist in the module."""
        from clarans import _clarans

        self.assertFalse(
            hasattr(_clarans, "_DELTA_TOL"),
            "_DELTA_TOL dead code should be removed"
        )


# ============================================================================
# Bug #11 Fix: _compute_2min uses argpartition
# ============================================================================
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
        rng = np.random.RandomState(42)
        X, _ = make_blobs(n_samples=50, centers=5, random_state=42)

        model = CLARANS(n_clusters=5, random_state=42)
        model.fit(X)

        # Manually compute expected 2-min using brute force
        medoids_dist = model._compute_medoids_distances(X, model.medoid_indices_)

        # Python fallback result
        with patch("clarans._clarans._core", None):
            near_idx, near_dist, second_dist = model._compute_2min(medoids_dist)

        # Brute force verification
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


# ============================================================================
# Bug #13 Fix: Thread-safe warning flag
# ============================================================================
class TestBug13Fix_ThreadSafeWarning(unittest.TestCase):
    """Verify that warning mechanism uses double-checked locking."""

    def test_warning_uses_lock(self):
        """Verify threading.Lock is used for warning flag."""
        from clarans import utils

        self.assertTrue(
            hasattr(utils, "_cython_warning_lock"),
            "Should have a threading.Lock for warning synchronization"
        )
        self.assertIsInstance(
            utils._cython_warning_lock,
            type(threading.Lock()),
            "_cython_warning_lock should be a Lock instance"
        )

    def test_double_checked_locking_pattern(self):
        """Source should use double-checked locking pattern."""
        import inspect
        source = inspect.getsource(
            __import__("clarans.utils", fromlist=["_warn_cython_unavailable"])._warn_cython_unavailable
        )

        # Double-checked locking: check flag, acquire lock, check flag again
        self.assertIn("_cython_warning_lock", source, "Should reference the lock")
        # Count occurrences of the flag check
        flag_checks = source.count("_cython_warning_issued")
        self.assertGreaterEqual(
            flag_checks, 2,
            "Should check _cython_warning_issued at least twice (double-checked locking)"
        )


# ============================================================================
# Regression: Ensure fixes don't break existing functionality
# ============================================================================
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
                model = ModelClass(n_clusters=3, init=init, num_local=1,
                                   max_neighbors=50, random_state=42)
                model.fit(X)
                self.assertEqual(
                    len(model.medoid_indices_), 3,
                    f"{ModelClass.__name__} init={init} failed"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
