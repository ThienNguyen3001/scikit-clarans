# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
CLARANS & FastCLARANS High-Performance C/Cython Kernels.
All core hotspots optimized with zero-allocation, single-pass algorithms.
"""

import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t


# ===========================================================================
# 1. CLARANS: Delta cost for a single medoid swap (Point 1)
# ===========================================================================
def clarans_delta(
    const ITYPE_t[::1] near_idx_map,
    const DTYPE_t[::1] near_dist,
    const DTYPE_t[::1] second_dist,
    const DTYPE_t[::1] d_xc,
    int random_medoid_pos,
    int n_samples
):
    """
    Compute CLARANS delta cost for swapping a single medoid in a single O(n) pass.
    Replaces multiple NumPy boolean masks and array allocations.
    """
    cdef DTYPE_t total_delta = 0.0
    cdef int j
    cdef DTYPE_t d, diff, val

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
    const ITYPE_t[::1] near_idx_map,
    const DTYPE_t[::1] near_dist,
    const DTYPE_t[::1] second_dist,
    const DTYPE_t[::1] d_xc,
    int n_samples,
    int n_clusters
):
    """
    Compute FastPAM1 delta cost for all k clusters in a single O(n) pass.
    Replaces 3 np.bincount calls and 3 boolean masks with a single pass.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] total_delta_np = np.zeros(n_clusters, dtype=np.float64)
    cdef DTYPE_t[::1] delta_arr = total_delta_np
    cdef DTYPE_t delta_td = 0.0
    cdef int j, m, best_m = 0
    cdef DTYPE_t d1, d2, dc, best_val

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
    const DTYPE_t[:, ::1] subD,
    int n_samples,
    int n_clusters
):
    """
    Find nearest and second-nearest medoid indices and distances in O(n*k).
    Replaces np.argsort(subD, axis=1) which is O(n*k*log(k)) and allocates (n, k) index array.
    """
    cdef cnp.ndarray[ITYPE_t, ndim=1] near_idx_map_np = np.empty(n_samples, dtype=np.int64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] near_dist_np = np.empty(n_samples, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] second_dist_np = np.empty(n_samples, dtype=np.float64)

    cdef ITYPE_t[::1] near_idx_map = near_idx_map_np
    cdef DTYPE_t[::1] near_dist = near_dist_np
    cdef DTYPE_t[::1] second_dist = second_dist_np

    cdef int i, m
    cdef DTYPE_t d, m1_val, m2_val
    cdef ITYPE_t m1_idx, m2_idx

    if n_clusters < 2:
        for i in range(n_samples):
            near_idx_map[i] = 0
            near_dist[i] = subD[i, 0]
            second_dist[i] = np.inf
        return near_idx_map_np, near_dist_np, second_dist_np

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
    const DTYPE_t[:, ::1] D,
    const ITYPE_t[::1] candidate_indices,
    const DTYPE_t[::1] dist_to_nearest,
    int n_samples,
    int n_candidates
):
    """
    Compute gains for all candidates in PAM BUILD in a memory-efficient C loop.
    Avoids allocating the large (n, n_candidates) diffs matrix in Python.
    """
    cdef int best_idx_in_cand = 0
    cdef DTYPE_t max_gain = -1.0
    cdef int c, i, cand
    cdef DTYPE_t gain, diff

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
    const DTYPE_t[::1] closest_dist_sq,
    const DTYPE_t[:, ::1] dists_candidates,
    const ITYPE_t[::1] candidate_ids,
    const ITYPE_t[::1] current_medoids,
    int n_samples,
    int n_local_trials,
    int n_current_medoids
):
    """
    Fuses minimum reduction and sum potential calculation for k-medoids++ trials.
    Avoids allocating intermediate candidate distance squared arrays in Python.
    """
    cdef int best_cand = -1
    cdef DTYPE_t best_pot = -1.0
    cdef cnp.ndarray[DTYPE_t, ndim=1] best_dist_sq = np.empty(n_samples, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] temp_dist_sq = np.empty(n_samples, dtype=np.float64)
    cdef DTYPE_t[::1] best_v = best_dist_sq
    cdef DTYPE_t[::1] temp_v = temp_dist_sq

    cdef int t, i, m, cand_id, in_medoids
    cdef DTYPE_t pot, val

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
    const DTYPE_t[:, ::1] X,
    const ITYPE_t[::1] medoid_indices,
    int n_samples,
    int n_features,
    int n_clusters
):
    """
    Calculate total clustering cost in 1 pass without allocating distance arrays.
    """
    cdef DTYPE_t total_cost = 0.0
    cdef int i, m, f, med_idx
    cdef DTYPE_t min_dist, dist, diff

    for i in range(n_samples):
        min_dist = 1e308
        for m in range(n_clusters):
            med_idx = medoid_indices[m]
            dist = 0.0
            for f in range(n_features):
                diff = X[i, f] - X[med_idx, f]
                dist += diff * diff
            dist = dist ** 0.5
            if dist < min_dist:
                min_dist = dist
        total_cost += min_dist

    return total_cost


# ===========================================================================
# 7. Single-row Euclidean distance on-the-fly (Point 7)
# ===========================================================================
def euclidean_distance_1_vs_n(
    const DTYPE_t[::1] cand_row,
    const DTYPE_t[:, ::1] X,
    cnp.ndarray[DTYPE_t, ndim=1] out_d_xc,
    int n_samples,
    int n_features
):
    """
    Compute Euclidean distance between a single candidate vector and all rows of X.
    Writes in-place into pre-allocated out_d_xc buffer, avoiding sklearn dispatch overhead.
    """
    cdef int i, f
    cdef DTYPE_t dist, diff
    cdef DTYPE_t[::1] out_v = out_d_xc

    for i in range(n_samples):
        dist = 0.0
        for f in range(n_features):
            diff = X[i, f] - cand_row[f]
            dist += diff * diff
        out_v[i] = dist ** 0.5


# ===========================================================================
# 8. O(1) Candidate Pool Tracking (Point 8)
# ===========================================================================
def swap_candidate_pool(
    cnp.ndarray[ITYPE_t, ndim=1] candidates,
    cnp.ndarray[ITYPE_t, ndim=1] pos_in_candidates,
    int old_medoid,
    int new_medoid
):
    """
    Swap old_medoid back into candidates and remove new_medoid in O(1).
    Avoids O(n) scan and memory reallocation of np.flatnonzero(non_medoid_mask).
    """
    cdef ITYPE_t[::1] c_view = candidates
    cdef ITYPE_t[::1] p_view = pos_in_candidates
    cdef int pos = p_view[new_medoid]

    c_view[pos] = old_medoid
    p_view[old_medoid] = pos
    p_view[new_medoid] = -1
