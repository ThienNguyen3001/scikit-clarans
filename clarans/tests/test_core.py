"""
Direct unit tests for CLARANS & FastCLARANS Cython core extensions (_core).

These tests evaluate the C-accelerated kernels directly against exact NumPy
reference implementations, verifying numerical precision for both float64 and
float32 fused types, 0-allocation behavior, and index handling.
"""

import unittest
import numpy as np

from clarans import HAS_CYTHON

try:
    from clarans import _core
except ImportError:
    _core = None  # type: ignore[assignment]


@unittest.skipUnless(
    HAS_CYTHON and _core is not None,
    "Cython extension '_core' not compiled or available",
)
class TestCythonCore(unittest.TestCase):
    def setUp(self):
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

    def test_clarans_delta_float64(self):
        n_samples = 200
        n_clusters = 5
        near_idx_map = self.rng.randint(0, n_clusters, size=n_samples).astype(np.intp)
        near_dist = self.rng.uniform(0.1, 5.0, size=n_samples).astype(np.float64)
        second_dist = near_dist + self.rng.uniform(0.1, 3.0, size=n_samples).astype(np.float64)
        d_xc = self.rng.uniform(0.1, 8.0, size=n_samples).astype(np.float64)

        for random_medoid_pos in range(n_clusters):
            py_delta = self._numpy_clarans_delta(
                near_idx_map, near_dist, second_dist, d_xc, random_medoid_pos
            )
            cy_delta = _core.clarans_delta(
                near_idx_map, near_dist, second_dist, d_xc, random_medoid_pos, n_samples
            )
            self.assertAlmostEqual(py_delta, cy_delta, places=10)

    def test_clarans_delta_float32(self):
        n_samples = 150
        n_clusters = 4
        near_idx_map = self.rng.randint(0, n_clusters, size=n_samples).astype(np.intp)
        near_dist = self.rng.uniform(0.1, 5.0, size=n_samples).astype(np.float32)
        second_dist = near_dist + self.rng.uniform(0.1, 3.0, size=n_samples).astype(np.float32)
        d_xc = self.rng.uniform(0.1, 8.0, size=n_samples).astype(np.float32)

        for random_medoid_pos in range(n_clusters):
            py_delta = self._numpy_clarans_delta(
                near_idx_map, near_dist, second_dist, d_xc, random_medoid_pos
            )
            cy_delta = _core.clarans_delta(
                near_idx_map, near_dist, second_dist, d_xc, random_medoid_pos, n_samples
            )
            self.assertAlmostEqual(py_delta, cy_delta, places=4)

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

    def test_fastpam1_delta_float64(self):
        n_samples = 300
        n_clusters = 6
        near_idx_map = self.rng.randint(0, n_clusters, size=n_samples).astype(np.intp)
        near_dist = self.rng.uniform(0.5, 4.0, size=n_samples).astype(np.float64)
        second_dist = near_dist + self.rng.uniform(0.2, 3.0, size=n_samples).astype(np.float64)
        d_xc = self.rng.uniform(0.1, 6.0, size=n_samples).astype(np.float64)

        py_best_m, py_best_val, py_delta_arr = self._numpy_fastpam1_delta(
            near_idx_map, near_dist, second_dist, d_xc, n_clusters
        )
        cy_best_m, cy_best_val, cy_delta_arr = _core.fastpam1_delta(
            near_idx_map, near_dist, second_dist, d_xc, n_samples, n_clusters
        )

        self.assertEqual(py_best_m, cy_best_m)
        self.assertAlmostEqual(py_best_val, cy_best_val, places=10)
        np.testing.assert_allclose(py_delta_arr, cy_delta_arr, rtol=1e-10)

    def test_fastpam1_delta_float32(self):
        n_samples = 200
        n_clusters = 4
        near_idx_map = self.rng.randint(0, n_clusters, size=n_samples).astype(np.intp)
        near_dist = self.rng.uniform(0.5, 4.0, size=n_samples).astype(np.float32)
        second_dist = near_dist + self.rng.uniform(0.2, 3.0, size=n_samples).astype(np.float32)
        d_xc = self.rng.uniform(0.1, 6.0, size=n_samples).astype(np.float32)

        py_best_m, py_best_val, py_delta_arr = self._numpy_fastpam1_delta(
            near_idx_map, near_dist, second_dist, d_xc, n_clusters
        )
        cy_best_m, cy_best_val, cy_delta_arr = _core.fastpam1_delta(
            near_idx_map, near_dist, second_dist, d_xc, n_samples, n_clusters
        )

        self.assertEqual(py_best_m, cy_best_m)
        self.assertAlmostEqual(py_best_val, cy_best_val, places=4)
        np.testing.assert_allclose(py_delta_arr, cy_delta_arr, rtol=1e-4)

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

        self.assertEqual(cy_m, cy_m_nobuf)
        self.assertAlmostEqual(cy_val, cy_val_nobuf, places=12)
        np.testing.assert_allclose(cy_arr, cy_arr_nobuf, rtol=1e-12)
        np.testing.assert_allclose(buf, cy_arr_nobuf, rtol=1e-12)

    # -----------------------------------------------------------------------
    # 3. update_cache_2min vs NumPy
    # -----------------------------------------------------------------------
    def test_update_cache_2min_general(self):
        n_samples = 150
        for n_clusters in [2, 3, 7]:
            subD = self.rng.uniform(0.5, 20.0, size=(n_samples, n_clusters)).astype(np.float64)
            # Ensure C-contiguous
            subD = np.ascontiguousarray(subD)

            near_idx, near_d, second_d = _core.update_cache_2min(subD, n_samples, n_clusters)

            sorted_indices = np.argsort(subD, axis=1)
            ref_near_idx = sorted_indices[:, 0]
            ref_near_d = np.take_along_axis(subD, sorted_indices[:, :1], axis=1).ravel()
            ref_second_d = np.take_along_axis(subD, sorted_indices[:, 1:2], axis=1).ravel()

            np.testing.assert_allclose(near_d, ref_near_d)
            np.testing.assert_allclose(second_d, ref_second_d)
            # When distances are strictly distinct, indices must match
            np.testing.assert_array_equal(near_idx, ref_near_idx)

    def test_update_cache_2min_k1(self):
        n_samples = 50
        n_clusters = 1
        subD = self.rng.uniform(0.5, 10.0, size=(n_samples, n_clusters)).astype(np.float64)
        subD = np.ascontiguousarray(subD)

        near_idx, near_d, second_d = _core.update_cache_2min(subD, n_samples, n_clusters)
        np.testing.assert_array_equal(near_idx, np.zeros(n_samples, dtype=np.intp))
        np.testing.assert_allclose(near_d, subD[:, 0])
        self.assertTrue(np.all(np.isinf(second_d)))

    def test_update_cache_2min_float32(self):
        n_samples = 80
        n_clusters = 3
        subD = self.rng.uniform(0.5, 15.0, size=(n_samples, n_clusters)).astype(np.float32)
        subD = np.ascontiguousarray(subD)

        near_idx, near_d, second_d = _core.update_cache_2min(subD, n_samples, n_clusters)
        sorted_indices = np.argsort(subD, axis=1)
        ref_near_d = np.take_along_axis(subD, sorted_indices[:, :1], axis=1).ravel()
        ref_second_d = np.take_along_axis(subD, sorted_indices[:, 1:2], axis=1).ravel()

        np.testing.assert_allclose(near_d, ref_near_d, rtol=1e-5)
        np.testing.assert_allclose(second_d, ref_second_d, rtol=1e-5)

    # -----------------------------------------------------------------------
    # 4. pam_build_step vs NumPy
    # -----------------------------------------------------------------------
    def test_pam_build_step(self):
        n_samples = 100
        n_candidates = 20
        D = self.rng.uniform(0.1, 10.0, size=(n_samples, n_samples)).astype(np.float64)
        D = np.ascontiguousarray(0.5 * (D + D.T))
        np.fill_diagonal(D, 0.0)

        dist_to_nearest = self.rng.uniform(2.0, 8.0, size=n_samples).astype(np.float64)
        candidate_indices = np.ascontiguousarray(
            self.rng.choice(n_samples, size=n_candidates, replace=False).astype(np.intp)
        )

        # NumPy reference
        diffs = np.maximum(0.0, dist_to_nearest[:, None] - D[:, candidate_indices])
        gains = np.sum(diffs, axis=0)
        ref_best_idx = int(np.argmax(gains))
        ref_max_gain = float(gains[ref_best_idx])

        cy_best_idx, cy_max_gain = _core.pam_build_step(
            D, candidate_indices, dist_to_nearest, n_samples, n_candidates
        )

        self.assertEqual(ref_best_idx, cy_best_idx)
        self.assertAlmostEqual(ref_max_gain, cy_max_gain, places=10)

    # -----------------------------------------------------------------------
    # 5. kmedoids_pp_trials vs NumPy
    # -----------------------------------------------------------------------
    def test_kmedoids_pp_trials(self):
        n_samples = 120
        n_trials = 10
        n_current = 2

        closest_dist_sq = self.rng.uniform(1.0, 10.0, size=n_samples).astype(np.float64)
        dists_candidates = np.ascontiguousarray(
            self.rng.uniform(0.5, 12.0, size=(n_trials, n_samples)).astype(np.float64)
        )
        candidate_ids = np.ascontiguousarray(np.arange(10, 10 + n_trials, dtype=np.intp))
        current_medoids = np.ascontiguousarray(np.array([2, 5], dtype=np.intp))

        # NumPy reference
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

        self.assertEqual(ref_best_cand, cy_best_cand)
        self.assertAlmostEqual(ref_best_pot, cy_best_pot, places=10)
        np.testing.assert_allclose(ref_best_dist_sq, cy_best_dist_sq, rtol=1e-10)

    # -----------------------------------------------------------------------
    # 6. is_matrix_symmetric vs NumPy
    # -----------------------------------------------------------------------
    def test_is_matrix_symmetric_float64_and_float32(self):
        for dtype in (np.float64, np.float32):
            n_samples = 50
            mat = self.rng.uniform(0.1, 10.0, size=(n_samples, n_samples)).astype(dtype)
            sym_mat = np.ascontiguousarray(0.5 * (mat + mat.T))

            # Strictly symmetric
            self.assertTrue(_core.is_matrix_symmetric(sym_mat, n_samples, 1e-6))

            # Asymmetric perturbation
            asym_mat = sym_mat.copy()
            asym_mat[10, 20] += 0.5
            self.assertFalse(_core.is_matrix_symmetric(asym_mat, n_samples, 1e-6))

            # Perturbation below tolerance
            near_sym = sym_mat.copy()
            near_sym[10, 20] += 1e-8
            near_sym[20, 10] -= 1e-8
            self.assertTrue(_core.is_matrix_symmetric(near_sym, n_samples, 1e-6))


if __name__ == "__main__":
    unittest.main()
