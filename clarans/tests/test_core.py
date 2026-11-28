"""
Direct unit tests for CLARANS & FastCLARANS Cython core extensions (_core).

These tests evaluate the C-accelerated kernels directly against exact NumPy
reference implementations, verifying numerical precision for both float64 and
float32 fused types, 0-allocation behavior, and index handling.
"""

import numpy as np
import pytest

from clarans import HAS_CYTHON

try:
    from clarans import _core
except ImportError:
    _core = None  # type: ignore[assignment]


@pytest.mark.skipif(
    not (HAS_CYTHON and _core is not None),
    reason="Cython extension '_core' not compiled or available",
)
class TestCythonCore:
    @pytest.fixture(autouse=True)
    def _setup_rng(self):
        self.rng = np.random.RandomState(42)

    # -----------------------------------------------------------------------
    # 1. clarans_delta vs NumPy
    # -----------------------------------------------------------------------
    def _numpy_clarans_delta(self, near_idx_map, near_dist, second_dist, d_xc, random_medoid_pos):
        mask1 = near_idx_map == random_medoid_pos
        delta1 = np.sum(np.minimum(second_dist[mask1], d_xc[mask1]) - near_dist[mask1])
        mask2 = (~mask1) & (d_xc < near_dist)
        delta2 = np.sum(d_xc[mask2] - near_dist[mask2])
        return float(delta1 + delta2)

    @pytest.mark.parametrize(
        "dtype, n_samples, n_clusters, places",
        [
            (np.float64, 200, 5, 10),
            (np.float32, 150, 4, 4),
        ],
    )
    def test_clarans_delta(self, dtype, n_samples, n_clusters, places):
        near_idx_map = self.rng.randint(0, n_clusters, size=n_samples).astype(np.intp)
        near_dist = self.rng.uniform(0.1, 5.0, size=n_samples).astype(dtype)
        second_dist = near_dist + self.rng.uniform(0.1, 3.0, size=n_samples).astype(dtype)
        d_xc = self.rng.uniform(0.1, 8.0, size=n_samples).astype(dtype)

        for random_medoid_pos in range(n_clusters):
            py_delta = self._numpy_clarans_delta(
                near_idx_map, near_dist, second_dist, d_xc, random_medoid_pos
            )
            cy_delta = _core.clarans_delta(
                near_idx_map, near_dist, second_dist, d_xc, random_medoid_pos, n_samples
            )
            assert np.isclose(py_delta, cy_delta, atol=10 ** (-places))

    # -----------------------------------------------------------------------
    # 2. fastpam1_delta vs NumPy
    # -----------------------------------------------------------------------
    def _numpy_fastpam1_delta(self, near_idx_map, near_dist, second_dist, d_xc, n_clusters):
        delta_arr = np.zeros(n_clusters, dtype=d_xc.dtype)
        mask_closer = d_xc < near_dist
        delta_td = np.sum(d_xc[mask_closer] - near_dist[mask_closer])

        mask_middle = (~mask_closer) & (d_xc < second_dist)
        diff_middle = d_xc[mask_middle] - near_dist[mask_middle]
        np.add.at(delta_arr, near_idx_map[mask_middle], diff_middle)

        mask_outer = (~mask_closer) & (~mask_middle)
        diff_outer = second_dist[mask_outer] - near_dist[mask_outer]
        np.add.at(delta_arr, near_idx_map[mask_outer], diff_outer)

        delta_arr += delta_td
        best_m = int(np.argmin(delta_arr))
        best_val = float(delta_arr[best_m])
        return best_m, best_val, delta_arr

    @pytest.mark.parametrize(
        "dtype, n_samples, n_clusters, places, rtol",
        [
            (np.float64, 300, 6, 10, 1e-10),
            (np.float32, 200, 4, 4, 1e-4),
        ],
    )
    def test_fastpam1_delta(self, dtype, n_samples, n_clusters, places, rtol):
        near_idx_map = self.rng.randint(0, n_clusters, size=n_samples).astype(np.intp)
        near_dist = self.rng.uniform(0.5, 4.0, size=n_samples).astype(dtype)
        second_dist = near_dist + self.rng.uniform(0.2, 3.0, size=n_samples).astype(dtype)
        d_xc = self.rng.uniform(0.1, 6.0, size=n_samples).astype(dtype)

        py_best_m, py_best_val, py_delta_arr = self._numpy_fastpam1_delta(
            near_idx_map, near_dist, second_dist, d_xc, n_clusters
        )
        cy_best_m, cy_best_val, cy_delta_arr = _core.fastpam1_delta(
            near_idx_map, near_dist, second_dist, d_xc, n_samples, n_clusters
        )

        assert py_best_m == cy_best_m
        assert np.isclose(py_best_val, cy_best_val, atol=10 ** (-places))
        np.testing.assert_allclose(py_delta_arr, cy_delta_arr, rtol=rtol)

    def test_fastpam1_delta_with_preallocated_buffer(self):
        n_samples = 250
        n_clusters = 5
        near_idx_map = self.rng.randint(0, n_clusters, size=n_samples).astype(np.intp)
        near_dist = self.rng.uniform(0.5, 4.0, size=n_samples).astype(np.float64)
        second_dist = near_dist + self.rng.uniform(0.2, 3.0, size=n_samples).astype(np.float64)
        d_xc = self.rng.uniform(0.1, 6.0, size=n_samples).astype(np.float64)

        buf = np.zeros(n_clusters, dtype=np.float64)
        cy_m, cy_val, cy_arr = _core.fastpam1_delta(
            near_idx_map, near_dist, second_dist, d_xc, n_samples, n_clusters, buf
        )
        cy_m_nobuf, cy_val_nobuf, cy_arr_nobuf = _core.fastpam1_delta(
            near_idx_map, near_dist, second_dist, d_xc, n_samples, n_clusters
        )

        assert cy_m == cy_m_nobuf
        assert np.isclose(cy_val, cy_val_nobuf, atol=1e-12)
        np.testing.assert_allclose(cy_arr, cy_arr_nobuf, rtol=1e-12)
        np.testing.assert_allclose(buf, cy_arr_nobuf, rtol=1e-12)

    # -----------------------------------------------------------------------
    # 3. update_cache_2min vs NumPy
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize(
        "dtype, rtol",
        [
            (np.float64, 1e-10),
            (np.float32, 1e-5),
        ],
    )
    def test_update_cache_2min(self, dtype, rtol):
        n_samples = 150
        for n_clusters in [2, 3, 7]:
            subD = self.rng.uniform(0.5, 20.0, size=(n_samples, n_clusters)).astype(dtype)
            subD = np.ascontiguousarray(subD)

            near_idx, near_d, second_d = _core.update_cache_2min(subD, n_samples, n_clusters)

            sorted_indices = np.argsort(subD, axis=1)
            ref_near_idx = sorted_indices[:, 0]
            ref_near_d = np.take_along_axis(subD, sorted_indices[:, :1], axis=1).ravel()
            ref_second_d = np.take_along_axis(subD, sorted_indices[:, 1:2], axis=1).ravel()

            np.testing.assert_allclose(near_d, ref_near_d, rtol=rtol)
            np.testing.assert_allclose(second_d, ref_second_d, rtol=rtol)
            np.testing.assert_array_equal(near_idx, ref_near_idx)

    def test_update_cache_2min_k1(self):
        n_samples = 50
        n_clusters = 1
        subD = self.rng.uniform(0.5, 10.0, size=(n_samples, n_clusters)).astype(np.float64)
        subD = np.ascontiguousarray(subD)

        near_idx, near_d, second_d = _core.update_cache_2min(subD, n_samples, n_clusters)
        np.testing.assert_array_equal(near_idx, np.zeros(n_samples, dtype=np.intp))
        np.testing.assert_allclose(near_d, subD[:, 0])
        assert np.all(np.isinf(second_d))

    def test_update_cache_2min_with_preallocated_buffers(self):
        n_samples = 200
        n_clusters = 5
        subD = self.rng.uniform(0.5, 20.0, size=(n_samples, n_clusters)).astype(np.float64)
        subD = np.ascontiguousarray(subD)

        idx_buf = np.empty(n_samples, dtype=np.intp)
        near_buf = np.empty(n_samples, dtype=np.float64)
        sec_buf = np.empty(n_samples, dtype=np.float64)

        near_idx, near_d, second_d = _core.update_cache_2min(
            subD, n_samples, n_clusters, idx_buf, near_buf, sec_buf, has_nan=0
        )
        assert near_idx is idx_buf
        assert near_d is near_buf
        assert second_d is sec_buf

        # Compare with no-buffer call
        ref_idx, ref_d, ref_sec = _core.update_cache_2min(subD, n_samples, n_clusters)
        np.testing.assert_array_equal(near_idx, ref_idx)
        np.testing.assert_allclose(near_d, ref_d, rtol=1e-12)
        np.testing.assert_allclose(second_d, ref_sec, rtol=1e-12)

        # Buffer shape validation
        with pytest.raises(ValueError, match="near_idx_map_buf length"):
            _core.update_cache_2min(
                subD, n_samples, n_clusters, near_idx_map_buf=np.empty(10, dtype=np.intp)
            )
        with pytest.raises(ValueError, match="near_dist_buf length"):
            _core.update_cache_2min(
                subD, n_samples, n_clusters, near_dist_buf=np.empty(10, dtype=np.float64)
            )
        with pytest.raises(ValueError, match="second_dist_buf length"):
            _core.update_cache_2min(
                subD, n_samples, n_clusters, second_dist_buf=np.empty(10, dtype=np.float64)
            )

    # -----------------------------------------------------------------------
    # 4. pam_build_step vs NumPy
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize(
        "dtype, places",
        [
            (np.float64, 10),
            (np.float32, 4),
        ],
    )
    def test_pam_build_step(self, dtype, places):
        n_samples = 100
        n_candidates = 20
        D = self.rng.uniform(0.1, 10.0, size=(n_samples, n_samples)).astype(dtype)
        D = np.ascontiguousarray(0.5 * (D + D.T))
        np.fill_diagonal(D, 0.0)

        dist_to_nearest = self.rng.uniform(2.0, 8.0, size=n_samples).astype(dtype)
        candidate_indices = np.ascontiguousarray(
            self.rng.choice(n_samples, size=n_candidates, replace=False).astype(np.intp)
        )

        diffs = np.maximum(0.0, dist_to_nearest[:, None] - D[:, candidate_indices])
        gains = np.sum(diffs, axis=0)
        ref_best_idx = int(np.argmax(gains))
        ref_max_gain = float(gains[ref_best_idx])

        cy_best_idx, cy_max_gain = _core.pam_build_step(
            D, candidate_indices, dist_to_nearest, n_samples, n_candidates
        )

        assert ref_best_idx == cy_best_idx
        assert np.isclose(ref_max_gain, cy_max_gain, atol=10 ** (-places))

    def test_pam_build_step_float32_double_accumulation(self):
        """pam_build_step must accumulate gains in float64 to avoid float32 cancellation drift."""
        n_samples = 20000
        dist_to_nearest = np.full(n_samples, 2.0, dtype=np.float32)
        dist_to_nearest[0] = np.float32(10002.0)

        D = np.full((n_samples, 3), 2.0, dtype=np.float32)
        # Candidate 1 gains 10000.0 at sample 0 plus ~5.9997 across remaining samples
        D[0, 1] = np.float32(2.0)
        D[1:, 1] = np.float32(1.9997)
        # Candidate 2 gains 10002.0 at sample 0 and 0 elsewhere (< Candidate 1 true gain)
        D[0, 2] = np.float32(0.0)
        D[1:, 2] = np.float32(2.0)

        candidate_indices = np.array([1, 2], dtype=np.intp)
        diffs64 = np.maximum(
            0.0,
            dist_to_nearest.astype(np.float64)[:, None]
            - D[:, candidate_indices].astype(np.float64),
        )
        gains64 = np.sum(diffs64, axis=0)
        ref_best_idx = int(np.argmax(gains64))
        ref_max_gain = float(gains64[ref_best_idx])

        cy_best_idx, cy_max_gain = _core.pam_build_step(
            D, candidate_indices, dist_to_nearest, n_samples, len(candidate_indices)
        )
        assert cy_best_idx == ref_best_idx == 0
        assert np.isclose(cy_max_gain, ref_max_gain, rtol=1e-6)

    # -----------------------------------------------------------------------
    # 5. kmedoids_pp_trials vs NumPy
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize(
        "dtype, places, rtol",
        [
            (np.float64, 10, 1e-10),
            (np.float32, 3, 1e-4),
        ],
    )
    def test_kmedoids_pp_trials(self, dtype, places, rtol):
        n_samples = 120
        n_trials = 10
        n_current = 2

        closest_dist_sq = self.rng.uniform(1.0, 10.0, size=n_samples).astype(dtype)
        dists_candidates = np.ascontiguousarray(
            self.rng.uniform(0.5, 12.0, size=(n_trials, n_samples)).astype(dtype)
        )
        candidate_ids = np.ascontiguousarray(np.arange(10, 10 + n_trials, dtype=np.intp))
        current_medoids = np.ascontiguousarray(np.array([2, 5], dtype=np.intp))

        ref_best_cand = -1
        ref_best_pot = np.inf
        ref_best_dist_sq = None
        for t in range(n_trials):
            cid = candidate_ids[t]
            if cid in current_medoids:
                continue
            new_dist_sq = np.minimum(closest_dist_sq, dists_candidates[t])
            pot = float(np.sum(new_dist_sq))
            if pot < ref_best_pot:
                ref_best_pot = pot
                ref_best_cand = cid
                ref_best_dist_sq = new_dist_sq

        cy_best_cand, cy_best_pot, cy_best_dist_sq = _core.kmedoids_pp_trials(
            closest_dist_sq,
            dists_candidates,
            candidate_ids,
            current_medoids,
            n_samples,
            n_trials,
            n_current,
        )

        assert ref_best_cand == cy_best_cand
        assert np.isclose(ref_best_pot, cy_best_pot, atol=10 ** (-places))
        np.testing.assert_allclose(ref_best_dist_sq, cy_best_dist_sq, rtol=rtol)

    def test_kmedoids_pp_trials_float32_double_accumulation(self):
        """kmedoids_pp_trials must accumulate potentials in float64 for float32 inputs."""
        n_samples = 20000
        closest_dist_sq = np.full(n_samples, 0.01, dtype=np.float32)
        closest_dist_sq[0] = np.float32(500000.0)

        dists_candidates = np.empty((2, n_samples), dtype=np.float32)
        # Trial 0 reduces remaining 19999 entries from 0.01 to 0.001 (~180 reduction)
        dists_candidates[0, 0] = np.float32(500000.0)
        dists_candidates[0, 1:] = np.float32(0.001)
        # Trial 1 only reduces entry 0 by 10.0 (far worse overall than Trial 0)
        dists_candidates[1, 0] = np.float32(499990.0)
        dists_candidates[1, 1:] = np.float32(0.01)

        candidate_ids = np.array([1, 2], dtype=np.intp)
        current_medoids = np.array([0], dtype=np.intp)

        ref_pot0 = float(np.sum(np.minimum(closest_dist_sq, dists_candidates[0]), dtype=np.float64))
        cy_best_cand, cy_best_pot, cy_best_dist_sq = _core.kmedoids_pp_trials(
            closest_dist_sq,
            dists_candidates,
            candidate_ids,
            current_medoids,
            n_samples,
            2,
            1,
        )
        assert cy_best_cand == 1
        assert np.isclose(cy_best_pot, ref_pot0, rtol=1e-7)
        assert cy_best_dist_sq.dtype == np.float32

    # -----------------------------------------------------------------------
    # 6. is_matrix_symmetric vs NumPy
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_is_matrix_symmetric(self, dtype):
        n_samples = 50
        mat = self.rng.uniform(0.1, 10.0, size=(n_samples, n_samples)).astype(dtype)
        sym_mat = np.ascontiguousarray(0.5 * (mat + mat.T))

        assert _core.is_matrix_symmetric(sym_mat, n_samples, 1e-6)

        asym_mat = sym_mat.copy()
        asym_mat[10, 20] += 0.5
        assert not _core.is_matrix_symmetric(asym_mat, n_samples, 1e-6)

        near_sym = sym_mat.copy()
        near_sym[10, 20] += 1e-8
        near_sym[20, 10] -= 1e-8
        assert _core.is_matrix_symmetric(near_sym, n_samples, 1e-6)

    def test_is_matrix_symmetric_rejects_diagonal_nan(self):
        """Diagonal NaNs must cause is_matrix_symmetric to return False like np.allclose."""
        for dtype in (np.float64, np.float32):
            mat = np.array([[np.nan, 1.0], [1.0, 0.0]], dtype=dtype)
            assert _core.is_matrix_symmetric(mat, 2, 1e-5, 1e-8) is False

    def test_kernels_validate_bounds_and_shapes(self):
        """All 6 Cython kernels must raise ValueError on invalid sizes or out-of-bounds indices."""
        idx_ok = np.array([0, 1, 0, 1], dtype=np.intp)
        vec_ok = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        sub_ok = np.ones((4, 2), dtype=np.float64)
        D_ok = np.ones((4, 4), dtype=np.float64)

        # 1. clarans_delta: negative n_samples or undersized vectors
        with pytest.raises(ValueError):
            _core.clarans_delta(idx_ok, vec_ok, vec_ok, vec_ok, 0, -1)
        with pytest.raises(ValueError):
            _core.clarans_delta(idx_ok[:2], vec_ok, vec_ok, vec_ok, 0, 4)

        # 2. fastpam1_delta: n_clusters < 1, undersized vectors, and out-of-bounds near_idx_map
        with pytest.raises(ValueError):
            _core.fastpam1_delta(idx_ok, vec_ok, vec_ok, vec_ok, 4, 0)
        with pytest.raises(ValueError):
            _core.fastpam1_delta(idx_ok[:2], vec_ok, vec_ok, vec_ok, 4, 2)
        bad_map = np.array([0, 3, 1, 0], dtype=np.intp)
        with pytest.raises(ValueError):
            _core.fastpam1_delta(bad_map, vec_ok, vec_ok, vec_ok, 4, 2)

        # 3. update_cache_2min: n_clusters < 1, n_samples < 0, subD too small
        with pytest.raises(ValueError):
            _core.update_cache_2min(sub_ok, 4, 0)
        with pytest.raises(ValueError):
            _core.update_cache_2min(sub_ok, 5, 2)
        with pytest.raises(ValueError):
            _core.update_cache_2min(sub_ok, 4, 3)

        # 4. pam_build_step: mismatched dimensions or out-of-bounds candidate index
        cand_ok = np.array([1, 2], dtype=np.intp)
        with pytest.raises(ValueError):
            _core.pam_build_step(D_ok, cand_ok, vec_ok, 5, 2)
        with pytest.raises(ValueError):
            _core.pam_build_step(D_ok, np.array([4], dtype=np.intp), vec_ok, 4, 1)
        with pytest.raises(ValueError):
            _core.pam_build_step(D_ok, np.array([-1], dtype=np.intp), vec_ok, 4, 1)

        # 5. kmedoids_pp_trials: mismatched array shapes
        cands_2d = np.ones((2, 4), dtype=np.float64)
        with pytest.raises(ValueError):
            _core.kmedoids_pp_trials(
                vec_ok[:2], cands_2d, cand_ok, np.array([0], dtype=np.intp), 4, 2, 1
            )
        with pytest.raises(ValueError):
            _core.kmedoids_pp_trials(
                vec_ok, cands_2d, cand_ok, np.array([0], dtype=np.intp), 4, 3, 1
            )

        # 6. is_matrix_symmetric: negative n_samples or D smaller than n_samples
        with pytest.raises(ValueError):
            _core.is_matrix_symmetric(D_ok, -1)
        with pytest.raises(ValueError):
            _core.is_matrix_symmetric(D_ok, 5)
