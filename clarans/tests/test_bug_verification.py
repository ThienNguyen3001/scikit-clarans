"""
Bug Report Verification Test Suite for scikit-clarans.

This module validates the correctness of each bug documented in `bug_report.md`.
It systematically executes rigorous test cases to determine whether each claim
in the report is:
  - [CONFIRMED BUG] An actual flaw in the implementation.
  - [CONFIRMED BEHAVIOR] A verified property or architectural trade-off.
  - [FALSE POSITIVE] An incorrect diagnosis where the code is actually correct.

Summary of Tested Items from bug_report.md:
  - Bug #1: _compute_2min NumPy fallback argpartition ordering
  - Bug #2: FastCLARANS pure Python FastPAM1 delta calculation
  - Bug #3: _compute_1_vs_n precomputed engine returns direct view (aliasing)
  - Bug #4: _precomputed_source memory leak on unhandled exception in fit()
  - Bug #5: Asymmetric distance matrix orientation
  - Bug #8: FastCLARANS get_params() exposure of cost_evaluation
  - Bug #10: Warning flag race condition / premature suppression
  - Bug #11: Type stub annotations in _core.pyi
"""

import ast
import inspect
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.metrics import pairwise_distances

from clarans import CLARANS, FastCLARANS, calculate_cost
from clarans.utils import HAS_CYTHON, EfficiencyWarning

try:
    from clarans import _core
except ImportError:
    _core = None  # type: ignore[assignment]


# ============================================================================
# Bug #1: _compute_2min NumPy fallback argpartition order
# ============================================================================
class TestBug1_ArgpartitionOrdering(unittest.TestCase):
    """Verifies Bug #1 from bug_report.md.

    CLAIM in bug_report.md:
        `np.argpartition(subD, 1, axis=1)[:, :2]` does NOT guarantee ordering
        between positions 0 and 1, so `part_idx[:, 0]` could be greater than
        `part_idx[:, 1]`, making `near_dist > second_dist` and ruining delta cost.

    VERIFICATION GOAL:
        Determine if `argpartition(arr, 1)` can ever place a larger element at
        index 0 than at index 1.
    """

    def test_argpartition_kth1_mathematical_invariant_exhaustive(self):
        """Mathematical invariant test: By definition of partition with kth=1,
        all elements strictly before kth (which is index 0 only) must be <= arr[kth].
        Therefore arr[part[0]] <= arr[part[1]] is mathematically guaranteed.
        """
        rng = np.random.RandomState(42)

        # Test over 50,000 diverse random arrays of varying sizes
        for _ in range(50000):
            k = rng.randint(2, 30)
            row = rng.randn(k)
            part = np.argpartition(row, 1)[:2]
            self.assertLessEqual(
                row[part[0]],
                row[part[1]],
                msg=f"VIOLATION: index 0 ({row[part[0]]}) > index 1 ({row[part[1]]}) for array {row}",
            )

    def test_argpartition_adversarial_patterns(self):
        """Test handcrafted adversarial patterns that might challenge partition
        algorithms: reverse-sorted, duplicates, extreme ranges, ties, infinities.
        """
        patterns = [
            [10.0, 9.0, 8.0, 7.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [5.0, 1.0, 3.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [100.0, 50.0],
            [0.0, 0.0, 1.0],
            [-10.0, -20.0, 0.0, 5.0],
            [1e15, 1e-15, 1.0, 0.0],
            [np.inf, 1.0, 2.0, 3.0],
            [-np.inf, np.inf, 0.0],
            [2.0, 2.0, 1.0, 1.0],
        ]
        for pattern in patterns:
            arr = np.array(pattern)
            part = np.argpartition(arr, 1)[:2]
            self.assertLessEqual(
                arr[part[0]],
                arr[part[1]],
                msg=f"Adversarial pattern failed: {pattern} -> partitioned: {arr[part]}",
            )

    def test_compute_2min_numpy_fallback_never_inverts_distances(self):
        """Test _compute_2min directly with _core patched to None (forcing NumPy fallback).
        near_dist must be <= second_dist for every single sample.
        """
        model = CLARANS(n_clusters=4, random_state=42)
        rng = np.random.RandomState(123)

        with patch("clarans._clarans._core", None):
            for _ in range(200):
                n_samples = rng.randint(10, 100)
                subD = rng.uniform(0.1, 100.0, size=(n_samples, 4))
                near_idx, near_d, second_d = model._compute_2min(subD)

                # Fundamental invariant: nearest distance <= second nearest distance
                diff = near_d - second_d
                self.assertTrue(
                    np.all(diff <= 1e-12),
                    f"NumPy fallback produced near_dist > second_dist: max diff = {np.max(diff)}",
                )

    @unittest.skipUnless(HAS_CYTHON and _core is not None, "Requires Cython _core")
    def test_compute_2min_numpy_fallback_matches_cython_exactly(self):
        """Compare Cython kernel update_cache_2min and NumPy fallback across
        multiple random subD matrices. Both must produce identical distances.
        """
        model = CLARANS(n_clusters=5, random_state=42)
        rng = np.random.RandomState(456)

        for _ in range(50):
            n_samples = 60
            subD = np.ascontiguousarray(rng.randn(n_samples, 5), dtype=np.float64)

            # Cython kernel
            c_near_idx, c_near_d, c_second_d = _core.update_cache_2min(subD, n_samples, 5)

            # NumPy fallback
            with patch("clarans._clarans._core", None):
                np_near_idx, np_near_d, np_second_d = model._compute_2min(subD)

            np.testing.assert_allclose(c_near_d, np_near_d, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(c_second_d, np_second_d, rtol=1e-14, atol=1e-14)


# ============================================================================
# Bug #2: FastCLARANS pure Python FastPAM1 fallback formula
# ============================================================================
class TestBug2_FastPAM1Formula(unittest.TestCase):
    """Verifies Bug #2 from bug_report.md (Retraction re-verification).

    CLAIM in bug_report.md:
        Initially claimed removal_loss and delta_td_plus_xc had double counting,
        then retracted because term1 and term2 properly compensate.

    VERIFICATION GOAL:
        Prove that FastCLARANS Python fallback path and Cython kernel produce
        identical delta calculations.
    """

    @unittest.skipUnless(HAS_CYTHON and _core is not None, "Requires Cython _core")
    def test_fastpam1_numpy_fallback_matches_cython_kernel(self):
        """Directly compare Cython fastpam1_delta with NumPy fallback formula."""
        rng = np.random.RandomState(42)
        n_samples = 80
        k = 4

        for _ in range(30):
            near_idx_map = rng.randint(0, k, size=n_samples).astype(np.intp)
            near_dist = rng.uniform(0.1, 5.0, size=n_samples).astype(np.float64)
            second_dist = near_dist + rng.uniform(0.1, 5.0, size=n_samples).astype(np.float64)
            d_xc = rng.uniform(0.0, 10.0, size=n_samples).astype(np.float64)

            # 1. Cython kernel
            c_best_m, c_min_delta, c_delta_arr = _core.fastpam1_delta(
                near_idx_map, near_dist, second_dist, d_xc, n_samples, k
            )

            # 2. Python fallback logic (as in _fast_clarans.py:350-388)
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

            # Assert identical results between Cython and pure Python
            np.testing.assert_allclose(c_delta_arr, total_delta, rtol=1e-10, atol=1e-10)
            self.assertEqual(c_best_m, py_best_m)
            self.assertAlmostEqual(c_min_delta, py_min_delta, places=9)


# ============================================================================
# Bug #3: _compute_1_vs_n precomputed engine returns direct view (aliasing)
# ============================================================================
class TestBug3_PrecomputedViewAliasing(unittest.TestCase):
    """Verifies Bug #3 from bug_report.md.

    CLAIM in bug_report.md:
        `_compute_1_vs_n` for precomputed engine returns `source[candidate_idx]`,
        which is a direct NumPy view of the underlying matrix rather than a copy.
        Mutating the returned array mutates the source matrix.

    VERIFICATION GOAL:
        Confirm whether returned vector shares memory with the input matrix.
    """

    def test_precomputed_symmetric_matrix_returns_view_sharing_memory(self):
        """For a symmetric C-contiguous distance matrix D, _compute_1_vs_n
        returns a view that shares memory with D.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3)
        D = pairwise_distances(X, metric="euclidean")
        D = np.ascontiguousarray(D, dtype=np.float64)

        model = CLARANS(n_clusters=3, metric="precomputed", random_state=42)
        model._setup_distance_engine(D)

        d_xc = model._compute_1_vs_n(None, D, candidate_idx=2)

        # CONFIRMED: d_xc shares memory with D
        self.assertTrue(
            np.shares_memory(d_xc, D),
            "d_xc should share memory with input matrix D when precomputed and symmetric",
        )
        self.assertIs(d_xc.base, D, "d_xc base should be D")

    def test_mutation_of_returned_view_mutates_source_matrix(self):
        """Demonstrates that in-place mutation of the returned view mutates
        the source distance matrix D.
        """
        rng = np.random.RandomState(42)
        D = pairwise_distances(rng.randn(10, 2))
        original_entry = D[3, 5]

        model = CLARANS(n_clusters=2, metric="precomputed", random_state=42)
        model._setup_distance_engine(D)

        d_xc = model._compute_1_vs_n(None, D, candidate_idx=3)
        d_xc[5] += 999.0

        # CONFIRMED: mutating d_xc mutates D[3, 5]
        self.assertEqual(
            D[3, 5],
            original_entry + 999.0,
            "Mutating d_xc view directly altered source matrix D",
        )

    def test_precomputed_engine_bypasses_out_buffer(self):
        """When out=d_xc_buf is passed to _compute_1_vs_n with precomputed engine,
        the buffer is ignored and the returned array is NOT the buffer.
        """
        D = pairwise_distances(np.random.RandomState(42).randn(15, 2))
        model = CLARANS(n_clusters=2, metric="precomputed", random_state=42)
        model._setup_distance_engine(D)

        buf = np.zeros(15, dtype=np.float64)
        result = model._compute_1_vs_n(None, D, candidate_idx=1, out=buf)

        self.assertIsNot(
            result,
            buf,
            "Precomputed engine should bypass out buffer (returns view instead of copying into out)",
        )
        self.assertTrue(
            np.all(buf == 0.0),
            "Buffer was left unmodified because precomputed engine ignored out parameter",
        )


# ============================================================================
# Bug #4: _precomputed_source memory leak on unhandled exception in fit()
# ============================================================================
class TestBug4_PrecomputedSourceMemoryLeak(unittest.TestCase):
    """Verifies Bug #4 from bug_report.md.

    CLAIM in bug_report.md:
        `self._precomputed_source` is set in `_setup_distance_engine(X)` and
        cleaned up in `_finalize_fit()`, but if an unhandled exception occurs
        during local search (e.g. keyboard interrupt, error in search),
        `_precomputed_source` remains attached to the model, leaking the full matrix.

    VERIFICATION GOAL:
        Confirm that an exception during local search leaves `_precomputed_source`
        dangling on both CLARANS and FastCLARANS instances.
    """

    def test_clarans_leaks_precomputed_source_on_unhandled_exception(self):
        """CLARANS: An unhandled exception during local search leaves _precomputed_source."""
        D = np.random.RandomState(42).rand(25, 25)
        model = CLARANS(n_clusters=3, metric="precomputed", random_state=42)

        def faulty_search(*args, **kwargs):
            raise RuntimeError("Unexpected failure during search")

        with patch.object(CLARANS, "_single_local_search", faulty_search):
            with self.assertRaises(RuntimeError):
                model.fit(D)

        # CONFIRMED BUG: _precomputed_source was NOT cleaned up!
        self.assertTrue(
            hasattr(model, "_precomputed_source"),
            "CONFIRMED BUG: _precomputed_source leaked on model after unhandled exception",
        )
        self.assertEqual(model._precomputed_source.shape, D.shape)

    def test_fastclarans_leaks_precomputed_source_on_unhandled_exception(self):
        """FastCLARANS: An unhandled exception during local search leaves _precomputed_source."""
        D = np.random.RandomState(42).rand(25, 25)
        model = FastCLARANS(n_clusters=3, metric="precomputed", random_state=42)

        def faulty_search(*args, **kwargs):
            raise RuntimeError("Unexpected failure during search")

        with patch.object(FastCLARANS, "_single_local_search", faulty_search):
            with self.assertRaises(RuntimeError):
                model.fit(D)

        # CONFIRMED BUG: _precomputed_source was NOT cleaned up!
        self.assertTrue(
            hasattr(model, "_precomputed_source"),
            "CONFIRMED BUG: FastCLARANS leaked _precomputed_source on unhandled exception",
        )


# ============================================================================
# Bug #5: Asymmetric distance matrix orientation
# ============================================================================
class TestBug5_AsymmetricPrecomputedOrientation(unittest.TestCase):
    """Verifies Bug #5 from bug_report.md (Retraction re-verification).

    CLAIM in bug_report.md:
        Initially suspected asymmetric matrix indexing D.T vs D was inverted,
        then retracted after inspecting _compute_medoids_distances and _compute_1_vs_n.

    VERIFICATION GOAL:
        Prove that asymmetric distance matrices produce inertia equal to
        manual calculate_cost.
    """

    def test_asymmetric_matrix_cost_invariance(self):
        """Inertia matches calculate_cost on an asymmetric distance matrix."""
        rng = np.random.RandomState(42)
        D = rng.uniform(0.1, 10.0, size=(20, 20))
        np.fill_diagonal(D, 0.0)

        for ModelClass in [CLARANS, FastCLARANS]:
            model = ModelClass(
                n_clusters=3,
                metric="precomputed",
                num_local=2,
                max_neighbors=50,
                random_state=42,
            ).fit(D)

            expected_cost = calculate_cost(D, model.medoid_indices_, metric="precomputed")
            self.assertAlmostEqual(
                model.inertia_,
                expected_cost,
                places=6,
                msg=f"{ModelClass.__name__} inertia mismatch on asymmetric matrix",
            )


# ============================================================================
# Bug #8: FastCLARANS get_params() exposure of cost_evaluation
# ============================================================================
class TestBug8_FastCLARANSApiConsistency(unittest.TestCase):
    """Verifies Bug #8 from bug_report.md.

    CLAIM in bug_report.md:
        `FastCLARANS` hardcodes `cost_evaluation="delta"` in super().__init__()
        and exposes the parameter via `get_params()` inherited from BaseEstimator.

    VERIFICATION GOAL:
        Check if `cost_evaluation` is in `get_params()`, and check how `set_params`
        and attribute access behave.
    """

    def test_fast_clarans_get_params_does_not_contain_cost_evaluation(self):
        """FastCLARANS.__init__ does NOT accept cost_evaluation in its signature.
        Therefore, scikit-learn's BaseEstimator.get_params() does NOT include it.
        (Contradicts the claim in bug_report.md: Bug #8 was a partial FALSE POSITIVE).
        """
        model = FastCLARANS(n_clusters=3)
        params = model.get_params()

        self.assertNotIn(
            "cost_evaluation",
            params,
            "cost_evaluation should NOT be in get_params() because __init__ omits it",
        )

    def test_fast_clarans_set_params_rejects_cost_evaluation(self):
        """set_params(cost_evaluation='brute_force') must be rejected with ValueError
        by scikit-learn BaseEstimator parameter validation.
        """
        model = FastCLARANS(n_clusters=3)
        with self.assertRaises(ValueError) as ctx:
            model.set_params(cost_evaluation="brute_force")
        self.assertIn("cost_evaluation", str(ctx.exception))

    def test_fast_clarans_has_cost_evaluation_attribute(self):
        """Even though omitted from get_params(), the attribute exists on the instance
        because of super().__init__(cost_evaluation='delta').
        """
        model = FastCLARANS(n_clusters=3)
        self.assertTrue(hasattr(model, "cost_evaluation"))
        self.assertEqual(model.cost_evaluation, "delta")


# ============================================================================
# Bug #10: Warning flag race condition / premature suppression
# ============================================================================
class TestBug10_WarningFlagOrder(unittest.TestCase):
    """Verifies Bug #10 from bug_report.md.

    CLAIM in bug_report.md:
        In `_warn_cython_unavailable()`, `_cython_warning_issued = True` is set
        BEFORE `warnings.warn(...)`. If `warnings.warn(...)` raises an exception
        (e.g., when filtered as error), the flag is already set, permanently
        suppressing the warning on future invocations.

    VERIFICATION GOAL:
        Demonstrate that raising an error on warning leaves the flag set to True.
    """

    def test_flag_set_before_warn_suppresses_subsequent_warnings(self):
        """When EfficiencyWarning is filtered as 'error', _warn_cython_unavailable
        raises the error. Because flag was set before warn, a subsequent call
        under 'always' filter will NOT issue the warning.
        """
        from clarans import utils

        # Save and reset state
        original_flag = utils._cython_warning_issued
        utils._cython_warning_issued = False

        try:
            with patch.object(utils, "HAS_CYTHON", False):
                # Filter warning as error
                with warnings.catch_warnings():
                    warnings.simplefilter("error", EfficiencyWarning)
                    with self.assertRaises(EfficiencyWarning):
                        utils._warn_cython_unavailable()

                # CONFIRMED: flag was set to True despite the exception
                self.assertTrue(
                    utils._cython_warning_issued,
                    "CONFIRMED: Flag was prematurely set to True before warnings.warn completed",
                )

                # Subsequent call with 'always' filter is suppressed
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always", EfficiencyWarning)
                    utils._warn_cython_unavailable()
                    self.assertEqual(
                        len(recorded),
                        0,
                        "Warning was suppressed on subsequent call because flag was prematurely set",
                    )
        finally:
            utils._cython_warning_issued = original_flag


# ============================================================================
# Bug #11: Type stub annotations in _core.pyi
# ============================================================================
class TestBug11_CoreTypeStub(unittest.TestCase):
    """Verifies Bug #11 from bug_report.md.

    CLAIM in bug_report.md:
        `_core.pyi` annotates `clarans_delta` as returning `float`, but at runtime
        Cython may return a numpy float scalar or python float depending on floating type.

    VERIFICATION GOAL:
        Inspect type stub syntax and runtime return type.
    """

    def test_pyi_stub_is_syntactically_valid(self):
        """Verify that _core.pyi can be parsed by Python's ast parser without error."""
        import pathlib
        pyi_path = pathlib.Path(__file__).parent.parent / "_core.pyi"
        self.assertTrue(pyi_path.exists(), "_core.pyi must exist")

        with open(pyi_path, "r", encoding="utf-8") as f:
            content = f.read()

        parsed = ast.parse(content)
        self.assertIsInstance(parsed, ast.Module)

    @unittest.skipUnless(HAS_CYTHON and _core is not None, "Requires Cython _core")
    def test_clarans_delta_runtime_return_type(self):
        """Verify actual runtime return type of clarans_delta."""
        near_idx = np.array([0, 0], dtype=np.intp)
        near_d = np.array([1.0, 1.0], dtype=np.float64)
        second_d = np.array([2.0, 2.0], dtype=np.float64)
        d_xc = np.array([0.5, 0.5], dtype=np.float64)

        result = _core.clarans_delta(near_idx, near_d, second_d, d_xc, 0, 2)
        # In Python runtime, floating return in Cython produces a Python float
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
