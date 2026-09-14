# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
"""
CLARANS & FastCLARANS High-Performance C/Cython Kernels.
All core hotspots optimized with zero-allocation, single-pass algorithms.

Conforming to scikit-learn Cython production standards:
- Fused types (floating: float, double)
- 64-bit safe indexing (Py_ssize_t, intp_t)
- Hardware sqrt intrinsics (libc.math.sqrt)
- OpenMP / multi-threading friendly (with nogil:)
- Direct typed memoryviews on all buffer arguments
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, INFINITY

# Initialize NumPy C API
cnp.import_array()

ctypedef cnp.intp_t intp_t

ctypedef fused floating:
    double
    float


# ===========================================================================
# 1. CLARANS: Delta cost for a single medoid swap (Point 1)
# ===========================================================================
def clarans_delta(
    const intp_t[::1] near_idx_map,
    const floating[::1] near_dist,
    const floating[::1] second_dist,
    const floating[::1] d_xc,
    intp_t random_medoid_pos,
    Py_ssize_t n_samples,
):
    """
    Compute CLARANS delta cost for swapping a single medoid in a single O(n) pass.
    Replaces multiple NumPy boolean masks and array allocations.
    """
    cdef:
        floating total_delta = 0.0
        Py_ssize_t j
        floating d, diff, val

    with nogil:
        for j in range(n_samples):
            d = d_xc[j]
            if near_idx_map[j] == random_medoid_pos:
                val = second_dist[j] if second_dist[j] < d else d
                total_delta += (val - near_dist[j])
            else:
                diff = d - near_dist[j]
                if diff < 0.0:
                    total_delta += diff

    return total_delta


# ===========================================================================
# 2. FastCLARANS: FastPAM1 delta cost for all k clusters (Point 2)
# ===========================================================================
def fastpam1_delta(
    const intp_t[::1] near_idx_map,
    const floating[::1] near_dist,
    const floating[::1] second_dist,
    const floating[::1] d_xc,
    Py_ssize_t n_samples,
    Py_ssize_t n_clusters,
):
    """
    Compute FastPAM1 delta cost for all k clusters in a single O(n) pass.
    Replaces 3 np.bincount calls and 3 boolean masks with a single pass.
    """
    cdef:
        cnp.ndarray[floating, ndim=1] total_delta_np = np.zeros(
            n_clusters, dtype=np.float64 if floating is double else np.float32
        )
        floating[::1] delta_arr = total_delta_np
        floating delta_td = 0.0
        Py_ssize_t j, m, best_m = 0
        floating d1, d2, dc, best_val

    with nogil:
        for j in range(n_samples):
            m = near_idx_map[j]
            d1 = near_dist[j]
            d2 = second_dist[j]
            dc = d_xc[j]

            if dc < d1:
                delta_td += (dc - d1)
            elif dc < d2:
                delta_arr[m] += (dc - d1)
            else:
                delta_arr[m] += (d2 - d1)

        best_val = delta_arr[0] + delta_td
        delta_arr[0] = best_val

        for m in range(1, n_clusters):
            delta_arr[m] += delta_td
            if delta_arr[m] < best_val:
                best_val = delta_arr[m]
                best_m = m

    return best_m, best_val, total_delta_np


# ===========================================================================
# 3. Cache: 2-nearest medoid linear scan (Point 3)
# ===========================================================================
def update_cache_2min(
    const floating[:, ::1] subD,
    Py_ssize_t n_samples,
    Py_ssize_t n_clusters,
):
    """
    Find nearest and second-nearest medoid indices and distances in O(n*k).
    Replaces np.argsort(subD, axis=1) which is O(n*k*log(k)) and allocates (n, k) index array.
    """
    cdef:
        cnp.ndarray[intp_t, ndim=1] near_idx_map_np = np.empty(n_samples, dtype=np.intp)
        cnp.ndarray[floating, ndim=1] near_dist_np = np.empty(
            n_samples, dtype=np.float64 if floating is double else np.float32
        )
        cnp.ndarray[floating, ndim=1] second_dist_np = np.empty(
            n_samples, dtype=np.float64 if floating is double else np.float32
        )

        intp_t[::1] near_idx_map = near_idx_map_np
        floating[::1] near_dist = near_dist_np
        floating[::1] second_dist = second_dist_np

        Py_ssize_t i, m
        floating d, m1_val, m2_val
        intp_t m1_idx, m2_idx

    if n_clusters < 2:
        with nogil:
            for i in range(n_samples):
                near_idx_map[i] = 0
                near_dist[i] = subD[i, 0]
                second_dist[i] = INFINITY
        return near_idx_map_np, near_dist_np, second_dist_np

    with nogil:
        for i in range(n_samples):
            if subD[i, 0] <= subD[i, 1]:
                m1_val = subD[i, 0]
                m1_idx = 0
                m2_val = subD[i, 1]
                m2_idx = 1
            else:
                m1_val = subD[i, 1]
                m1_idx = 1
                m2_val = subD[i, 0]
                m2_idx = 0

            for m in range(2, n_clusters):
                d = subD[i, m]
                if d < m1_val:
                    m2_val = m1_val
                    m2_idx = m1_idx
                    m1_val = d
                    m1_idx = m
                elif d < m2_val:
                    m2_val = d
                    m2_idx = m
            near_idx_map[i] = m1_idx
            near_dist[i] = m1_val
            second_dist[i] = m2_val

    return near_idx_map_np, near_dist_np, second_dist_np


# ===========================================================================
# 4. PAM BUILD greedy step (Point 4)
# ===========================================================================
def pam_build_step(
    const floating[:, ::1] D,
    const intp_t[::1] candidate_indices,
    const floating[::1] dist_to_nearest,
    Py_ssize_t n_samples,
    Py_ssize_t n_candidates,
):
    """
    Compute gains for all candidates in PAM BUILD in a memory-efficient C loop.
    Avoids allocating the large (n, n_candidates) diffs matrix in Python.
    """
    cdef:
        Py_ssize_t best_idx_in_cand = 0
        floating max_gain = -1.0
        Py_ssize_t c, i, cand
        floating gain, diff

    with nogil:
        for c in range(n_candidates):
            cand = candidate_indices[c]
            gain = 0.0
            for i in range(n_samples):
                diff = dist_to_nearest[i] - D[i, cand]
                if diff > 0.0:
                    gain += diff
            if gain > max_gain:
                max_gain = gain
                best_idx_in_cand = c

    return best_idx_in_cand, max_gain


# ===========================================================================
# 5. k-medoids++ seeding local trials (Point 5)
# ===========================================================================
def kmedoids_pp_trials(
    const floating[::1] closest_dist_sq,
    const floating[:, ::1] dists_candidates,
    const intp_t[::1] candidate_ids,
    const intp_t[::1] current_medoids,
    Py_ssize_t n_samples,
    Py_ssize_t n_local_trials,
    Py_ssize_t n_current_medoids,
):
    """
    Fuses minimum reduction and sum potential calculation for k-medoids++ trials.
    Avoids allocating intermediate candidate distance squared arrays in Python.
    """
    cdef:
        intp_t best_cand = -1
        floating best_pot = -1.0
        cnp.ndarray[floating, ndim=1] best_dist_sq = np.empty(
            n_samples, dtype=np.float64 if floating is double else np.float32
        )
        cnp.ndarray[floating, ndim=1] temp_dist_sq = np.empty(
            n_samples, dtype=np.float64 if floating is double else np.float32
        )
        floating[::1] best_v = best_dist_sq
        floating[::1] temp_v = temp_dist_sq

        Py_ssize_t t, i, m, cand_id
        int in_medoids
        floating pot, val

    with nogil:
        for t in range(n_local_trials):
            cand_id = candidate_ids[t]
            in_medoids = 0
            for m in range(n_current_medoids):
                if current_medoids[m] == cand_id:
                    in_medoids = 1
                    break
            if in_medoids:
                continue

            pot = 0.0
            for i in range(n_samples):
                val = closest_dist_sq[i]
                if dists_candidates[t, i] < val:
                    val = dists_candidates[t, i]
                temp_v[i] = val
                pot += val

            if best_pot < 0.0 or pot < best_pot:
                best_pot = pot
                best_cand = cand_id
                for i in range(n_samples):
                    best_v[i] = temp_v[i]

    return best_cand, best_pot, best_dist_sq


# ===========================================================================
# 6. Single-pass Euclidean cost calculation (Point 6)
# ===========================================================================
def euclidean_cost_1pass(
    const floating[:, ::1] X,
    const intp_t[::1] medoid_indices,
    Py_ssize_t n_samples,
    Py_ssize_t n_features,
    Py_ssize_t n_clusters,
):
    """
    Calculate total clustering cost in 1 pass without allocating distance arrays.
    """
    cdef:
        floating total_cost = 0.0
        Py_ssize_t i, m, f, med_idx
        floating min_dist, dist, diff

    with nogil:
        for i in range(n_samples):
            min_dist = 1e308 if floating is double else 1e38
            for m in range(n_clusters):
                med_idx = medoid_indices[m]
                dist = 0.0
                for f in range(n_features):
                    diff = X[i, f] - X[med_idx, f]
                    dist += diff * diff
                dist = sqrt(dist)
                if dist < min_dist:
                    min_dist = dist
            total_cost += min_dist

    return total_cost


# ===========================================================================
# 7. Single-row Euclidean distance on-the-fly (Point 7)
# ===========================================================================
def euclidean_distance_1_vs_n(
    const floating[::1] cand_row,
    const floating[:, ::1] X,
    floating[::1] out_d_xc,
    Py_ssize_t n_samples,
    Py_ssize_t n_features,
):
    """
    Compute Euclidean distance between a single candidate vector and all rows of X.
    Writes in-place into pre-allocated out_d_xc buffer, avoiding sklearn dispatch overhead.
    """
    cdef:
        Py_ssize_t i, f
        floating dist, diff

    with nogil:
        for i in range(n_samples):
            dist = 0.0
            for f in range(n_features):
                diff = X[i, f] - cand_row[f]
                dist += diff * diff
            out_d_xc[i] = sqrt(dist)


# ===========================================================================
# 8. O(1) Candidate Pool Tracking (Point 8)
# ===========================================================================
def swap_candidate_pool(
    intp_t[::1] candidates,
    intp_t[::1] pos_in_candidates,
    intp_t old_medoid,
    intp_t new_medoid,
):
    """
    Swap old_medoid back into candidates and remove new_medoid in O(1).
    Avoids O(n) scan and memory reallocation of np.flatnonzero(non_medoid_mask).
    """
    cdef Py_ssize_t pos

    with nogil:
        pos = pos_in_candidates[new_medoid]
        candidates[pos] = old_medoid
        pos_in_candidates[old_medoid] = pos
        pos_in_candidates[new_medoid] = -1
