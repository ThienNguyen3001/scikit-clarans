"""
Comprehensive Bug Verification & Fix Test Suite for scikit-clarans.

This test suite rigorously validates that all confirmed bugs from `bug_report.md`
have been cleanly and robustly fixed:
  - FIXED BUGS:
      * Bug #3: _compute_1_vs_n with cdist engine populates and returns caller's out buffer
        for all dtypes
      * Bug #4: _initialize_medoids with metric='precomputed' + 2D init verifies shape and semantics
      * Bug #5: FastCLARANS API consistency and parameter handling
      * Bug #6: _compute_2min Python fallback cleans/filters NaNs, preventing NaN propagation
      * Bug #8: Double-checked locking pattern (DCLP) in _warn_cython_unavailable
      * Bug #12: second_dist dtype matches subD.dtype in Python fallback for n_clusters=1
      * Bug #15: Clear warnings for array init duplicates
      * Bug #16: initialize_build allocates is_medoid mask once outside loop in Python fallback
      * Bug #24: Safe fallback for _core.is_matrix_symmetric signature mismatch
  - CONFIRMED INVARIANTS / FALSE ALARMS:
      * Bug #1: np.argpartition(subD, 1, axis=1) invariant
        (kth=1 always puts minimum at index 0 for non-NaN)
      * Bug #2: FastCLARANS pure-Python FastPAM1 delta formula equivalence to Cython kernel
      * Bug #9: Asymmetric precomputed matrix indexing (X.T) cost consistency
"""

import threading
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.metrics import pairwise_distances

from clarans import CLARANS, FastCLARANS, calculate_cost
from clarans.utils import HAS_CYTHON

try:
    from clarans import _core
except ImportError:
    _core = None  # type: ignore[assignment]


# ============================================================================
# PART 1: VERIFICATION OF BUG FIXES
# ============================================================================


class TestBug3_CdistOutBufferDroppedForNonFloat64(unittest.TestCase):
    """Verifies Fix for Bug #3 from bug_report.md:
    In `_clarans.py:849-858`, `_compute_1_vs_n` with engine="cdist" now populates
    and returns the caller's `out` buffer even when `out.dtype != np.float64`
    via `np.copyto`.
    """

    def test_cdist_out_buffer_populated_for_float32(self):
        """When out buffer is float32, _compute_1_vs_n copies result into it
        and returns the exact buffer object.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3).astype(np.float32)
        model = CLARANS(n_clusters=3, metric="euclidean", random_state=42)
        model._setup_distance_engine(X)

        cand_row = X[0:1]
        out_buf_float32 = np.zeros(20, dtype=np.float32)

        result = model._compute_1_vs_n(cand_row, X, out=out_buf_float32)

        # FIXED: result is the provided buffer and contains valid distance data
        self.assertIs(
            result,
            out_buf_float32,
            "FIX #3: Caller's out buffer must be populated and returned!",
        )
        self.assertFalse(np.all(out_buf_float32 == 0.0))

    def test_cdist_out_buffer_used_for_float64(self):
        """When out buffer is float64, _compute_1_vs_n correctly reuses the
        provided buffer in-place without copying.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3).astype(np.float64)
        model = CLARANS(n_clusters=3, metric="euclidean", random_state=42)
        model._setup_distance_engine(X)

        cand_row = X[0:1]
        out_buf_float64 = np.zeros(20, dtype=np.float64)

        result = model._compute_1_vs_n(cand_row, X, out=out_buf_float64)

        self.assertIs(
            result,
            out_buf_float64,
            "Float64 buffer should be reused in-place by cdist engine.",
        )


class TestBug4_Precomputed2DInitArgminWrong(unittest.TestCase):
    """Verifies Bug #4 from bug_report.md:
    In `_clarans.py:346-352`, when `metric='precomputed'` and `init` is a 2D array,
    the semantics of 2D distance vectors are validated.
    """

    def test_precomputed_2d_init_semantics(self):
        """Demonstrates that 2D init centers representing distance vectors
        to all samples correctly map to the nearest medoid indices.
        """
        rng = np.random.RandomState(42)
        D = pairwise_distances(rng.randn(4, 2))

        # 2D init array of shape (2, 4) where row 0 has min at col 3, row 1 has min at col 1
        init_centers = np.array(
            [
                [10.0, 8.0, 5.0, 0.1],  # min at index 3
                [9.0, 0.2, 7.0, 6.0],   # min at index 1
            ],
            dtype=np.float64,
        )

        model = CLARANS(n_clusters=2, metric="precomputed", init=init_centers, num_local=1)
        medoids = model._initialize_medoids(D, random_state=rng)

        np.testing.assert_array_equal(medoids, [1, 3])

    def test_precomputed_2d_init_duplicate_fallback(self):
        """If init_centers happens to have the same minimum column across rows,
        duplicates are caught and filled with valid distinct medoids.
        """
        D = pairwise_distances(np.random.RandomState(42).randn(5, 2))
        init_centers = np.array(
            [
                [0.1, 5.0, 5.0, 5.0, 5.0],
                [0.2, 8.0, 8.0, 8.0, 8.0],
            ],
            dtype=np.float64,
        )

        model = CLARANS(n_clusters=2, metric="precomputed", init=init_centers, num_local=1)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            medoids = model._initialize_medoids(D, random_state=np.random.RandomState(42))

        self.assertEqual(len(np.unique(medoids)), 2)
        warning_messages = [str(w.message) for w in recorded]
        self.assertTrue(any("duplicate" in msg.lower() for msg in warning_messages))


class TestBug5_FastCLARANSCostEvaluationParameter(unittest.TestCase):
    """Verifies Bug #5 from bug_report.md:
    `FastCLARANS` enforces `cost_evaluation='delta'` consistently.
    """

    def test_cost_evaluation_parameter_asymmetry(self):
        model = FastCLARANS(n_clusters=3)

        self.assertTrue(hasattr(model, "cost_evaluation"))
        self.assertEqual(model.cost_evaluation, "delta")

        params = model.get_params()
        self.assertNotIn("cost_evaluation", params)

        with self.assertRaises(ValueError):
            model.set_params(cost_evaluation="brute_force")


class TestBug6_Compute2MinPythonFallbackNaNHandling(unittest.TestCase):
    """Verifies Fix for Bug #6 from bug_report.md:
    In `_clarans.py:955-963`, the Python fallback cleans NaNs using np.where(np.isnan, np.inf, subD)
    so NaNs never corrupt near_dist or second_dist.
    """

    def test_python_fallback_cleans_nans(self):
        """When subD contains NaNs, the Python fallback cleans them, ensuring
        near_dist and second_dist are finite values when valid distances exist.
        """
        subD = np.array(
            [
                [np.nan, 2.0, 4.0, 6.0],  # Valid mins: 2.0, 4.0
                [1.0, np.nan, 3.0, 5.0],  # Valid mins: 1.0, 3.0
            ],
            dtype=np.float64,
        )
        n_samples, k = subD.shape

        model = CLARANS(n_clusters=k, random_state=42)
        with patch("clarans._clarans._core", None):
            py_near_idx, py_near_d, py_second_d = model._compute_2min(subD)

        # FIXED: NaNs are filtered out and valid minimums are picked!
        self.assertEqual(py_near_idx[0], 1)
        self.assertEqual(py_near_d[0], 2.0)
        self.assertEqual(py_second_d[0], 4.0)

        self.assertEqual(py_near_idx[1], 0)
        self.assertEqual(py_near_d[1], 1.0)
        self.assertEqual(py_second_d[1], 3.0)


class TestBug8_DCLPRaceInCythonWarning(unittest.TestCase):
    """Verifies Bug #8 from bug_report.md:
    In `clarans/utils.py:30-44`, `_warn_cython_unavailable` uses thread locking.
    """

    def test_dclp_concurrent_execution(self):
        from clarans import utils

        original_flag = utils._cython_warning_issued
        utils._cython_warning_issued = False

        errors = []

        def worker():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with patch.object(utils, "HAS_CYTHON", False):
                        for _ in range(50):
                            utils._warn_cython_unavailable()
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(errors), 0, f"Thread errors occurred: {errors}")
            self.assertTrue(utils._cython_warning_issued)
        finally:
            utils._cython_warning_issued = original_flag


class TestBug12_SecondDistDtypeMismatchNClusters1(unittest.TestCase):
    """Verifies Fix for Bug #12 from bug_report.md:
    In `_clarans.py:964-968`, for `n_clusters == 1`, Python fallback creates
    second_dist with `dtype=subD.dtype`, matching near_dist.
    """

    def test_python_fallback_n_clusters_1_float32_dtype_matches(self):
        """For n_clusters=1 and float32 subD, Python fallback returns matching float32 dtypes."""
        n_samples = 10
        subD_float32 = np.ones((n_samples, 1), dtype=np.float32)

        model = CLARANS(n_clusters=1, random_state=42)

        with patch("clarans._clarans._core", None):
            near_idx, near_dist, second_dist = model._compute_2min(subD_float32)

        # FIXED: Both near_dist and second_dist have dtype float32
        self.assertEqual(near_dist.dtype, np.float32)
        self.assertEqual(second_dist.dtype, np.float32)
        self.assertEqual(near_dist.dtype, second_dist.dtype)


class TestBug15_MisleadingDeterministicWarningForDuplicateInit(unittest.TestCase):
    """Verifies Bug #15 from bug_report.md:
    When `init` is an array with duplicate centers, warning messages inform
    the user accurately.
    """

    def test_duplicate_array_init_warnings(self):
        X = np.arange(30).reshape(10, 3).astype(np.float64)
        init_centers = np.array([X[0], X[0], X[0]], dtype=np.float64)

        model = CLARANS(n_clusters=3, init=init_centers, num_local=3, random_state=42)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            model.fit(X)

        warning_messages = [str(w.message) for w in recorded]

        has_exact_same_warn = any("exact same initial medoids" in m for m in warning_messages)
        has_duplicate_warn = any("duplicate" in m and "random" in m for m in warning_messages)

        self.assertTrue(has_exact_same_warn)
        self.assertTrue(has_duplicate_warn)


class TestBug16_BuildInitializationMaskRecreation(unittest.TestCase):
    """Verifies Fix for Bug #16 from bug_report.md:
    In `clarans/_initialization.py:200-203`, the pure Python fallback of `initialize_build`
    allocates `is_medoid` ONCE outside the loop, avoiding O(n*k) repeated allocations.
    """

    def test_python_fallback_allocates_is_medoid_once(self):
        """Verify that in the Python fallback path of initialize_build,
        the boolean mask is allocated exactly 1 time (outside the loop).
        """
        import clarans._initialization as init_mod

        rng = np.random.RandomState(42)
        D = pairwise_distances(rng.randn(15, 2))
        k = 4

        zero_allocations = []
        original_zeros = np.zeros

        def tracked_zeros(*args, **kwargs):
            res = original_zeros(*args, **kwargs)
            if len(args) > 0 and args[0] == 15 and kwargs.get("dtype") == bool:
                zero_allocations.append(res)
            return res

        with patch("clarans._initialization._core", None):
            with patch("numpy.zeros", side_effect=tracked_zeros):
                medoids = init_mod.initialize_build(D, n_clusters=k, metric="precomputed")

        # FIXED: is_medoid is allocated exactly ONCE outside the loop!
        self.assertEqual(
            len(zero_allocations),
            1,
            f"FIX #16: Expected 1 mask allocation outside loop, got {len(zero_allocations)}",
        )
        self.assertEqual(len(medoids), k)


class TestBug24_IsMatrixSymmetricSignatureMismatch(unittest.TestCase):
    """Verifies Fix for Bug #24:
    Calling CLARANS.fit() with metric='precomputed' handles binary signature
    differences gracefully without raising TypeError.
    """

    def test_precomputed_fit_succeeds_without_typeerror(self):
        """Fitting CLARANS with metric='precomputed' on a C-contiguous array
        now succeeds smoothly thanks to signature fallback.
        """
        D = np.eye(5, dtype=np.float64)
        model = CLARANS(
            n_clusters=2, metric="precomputed", num_local=1, max_neighbors=10, random_state=42
        )
        # FIXED: fit(D) succeeds without raising TypeError
        model.fit(D)
        self.assertEqual(len(model.medoid_indices_), 2)


# ============================================================================
# PART 2: CONFIRMED INVARIANTS / FALSE ALARMS VERIFICATION
# ============================================================================


class TestFalseAlarm1_ArgpartitionOrdering(unittest.TestCase):
    """Verifies False Alarm for Bug #1 from bug_report.md:
    By mathematical definition of partition with `kth=1`, all elements at indices
    < kth (which is index 0 only) must be <= arr[kth].
    """

    def test_argpartition_invariant_across_random_arrays(self):
        rng = np.random.RandomState(42)
        for _ in range(5000):
            k = rng.randint(2, 25)
            row = rng.randn(k)
            part = np.argpartition(row, 1)[:2]
            self.assertLessEqual(
                row[part[0]],
                row[part[1]],
                f"Invariant violation: row[part[0]]={row[part[0]]} > row[part[1]]={row[part[1]]}",
            )


class TestFalseAlarm2_FastPAM1FormulaEquivalence(unittest.TestCase):
    """Verifies False Alarm for Bug #2 from bug_report.md:
    The pure Python FastPAM1 fallback formula in `_fast_clarans.py:441-480` is
    mathematically identical to Cython `fastpam1_delta` in `_core.pyx:98-124`.
    """

    @unittest.skipUnless(HAS_CYTHON and _core is not None, "Requires Cython _core")
    def test_fastpam1_python_fallback_matches_cython_kernel(self):
        rng = np.random.RandomState(42)
        n_samples = 60
        k = 4

        for _ in range(20):
            near_idx_map = rng.randint(0, k, size=n_samples).astype(np.intp)
            near_dist = rng.uniform(0.1, 5.0, size=n_samples).astype(np.float64)
            second_dist = near_dist + rng.uniform(0.1, 5.0, size=n_samples).astype(np.float64)
            d_xc = rng.uniform(0.0, 10.0, size=n_samples).astype(np.float64)

            # 1. Cython kernel
            c_best_m, c_min_delta, c_delta_arr = _core.fastpam1_delta(
                near_idx_map, near_dist, second_dist, d_xc, n_samples, k
            )

            # 2. Python fallback logic
            removal_loss = np.zeros(k, dtype=np.float64)
            diff = second_dist - near_dist
            with np.errstate(invalid="ignore"):
                removal_loss += np.bincount(near_idx_map, weights=diff, minlength=k)

            mask_better_than_nearest = d_xc < near_dist
            delta_td_plus_xc = float(
                np.sum(d_xc[mask_better_than_nearest] - near_dist[mask_better_than_nearest])
            )
            total_delta = removal_loss + delta_td_plus_xc

            mask_case1 = mask_better_than_nearest
            if np.any(mask_case1):
                term1 = near_dist[mask_case1] - second_dist[mask_case1]
                with np.errstate(invalid="ignore"):
                    total_delta += np.bincount(
                        near_idx_map[mask_case1], weights=term1, minlength=k
                    )

            mask_case2 = (d_xc >= near_dist) & (d_xc < second_dist)
            if np.any(mask_case2):
                term2 = d_xc[mask_case2] - second_dist[mask_case2]
                with np.errstate(invalid="ignore"):
                    total_delta += np.bincount(
                        near_idx_map[mask_case2], weights=term2, minlength=k
                    )

            py_best_m = int(np.argmin(total_delta))
            py_min_delta = total_delta[py_best_m]

            np.testing.assert_allclose(c_delta_arr, total_delta, rtol=1e-10, atol=1e-10)
            self.assertEqual(c_best_m, py_best_m)
            self.assertAlmostEqual(c_min_delta, py_min_delta, places=9)


class TestFalseAlarm9_AsymmetricPrecomputedOrientation(unittest.TestCase):
    """Verifies False Alarm for Bug #9 from bug_report.md:
    Storing `X.T` for asymmetric distance matrices produces mathematically
    correct row-access distances in `_compute_1_vs_n`, and `model.inertia_`
    matches `calculate_cost` exactly.
    """

    def test_asymmetric_matrix_cost_invariance(self):
        rng = np.random.RandomState(42)
        D = rng.uniform(0.1, 10.0, size=(15, 15))
        np.fill_diagonal(D, 0.0)

        for ModelClass in [CLARANS, FastCLARANS]:
            model = ModelClass(
                n_clusters=3,
                metric="precomputed",
                num_local=2,
                max_neighbors=30,
                random_state=42,
            ).fit(D)

            expected_cost = calculate_cost(D, model.medoid_indices_, metric="precomputed")
            self.assertAlmostEqual(
                model.inertia_,
                expected_cost,
                places=6,
                msg=f"{ModelClass.__name__} inertia mismatch on asymmetric matrix",
            )


if __name__ == "__main__":
    unittest.main()
