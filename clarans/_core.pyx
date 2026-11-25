# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
"""Cython kernels for CLARANS and FastCLARANS delta and distance calculations."""


import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, INFINITY, isnan, isinf, isfinite, fabs
from libc.stdlib cimport malloc, free

# Initialize NumPy C API
cnp.import_array()

ctypedef cnp.intp_t intp_t

ctypedef fused floating:
    double
    float


# CLARANS single-swap delta calculation
def clarans_delta(
    const intp_t[::1] near_idx_map,
    const floating[::1] near_dist,
    const floating[::1] second_dist,
    const floating[::1] d_xc,
    intp_t random_medoid_pos,
    Py_ssize_t n_samples,
):
    """Compute CLARANS delta cost for a candidate medoid swap."""
    cdef:
        double total_delta = 0.0
        Py_ssize_t j
        double d, d1, d2, diff, val
        long long inf_delta = 0

    with nogil:
        for j in range(n_samples):
            d = <double>d_xc[j]
            if isnan(d):
                d = INFINITY
            if near_idx_map[j] == random_medoid_pos:
                val = <double>second_dist[j] if second_dist[j] < d else d
                total_delta += (val - <double>near_dist[j])
            else:
                diff = d - <double>near_dist[j]
                if diff < 0.0:
                    total_delta += diff

        if not isfinite(total_delta):
            total_delta = 0.0
            inf_delta = 0
            for j in range(n_samples):
                d = <double>d_xc[j]
                if isnan(d):
                    d = INFINITY
                d1 = <double>near_dist[j]
                if isnan(d1):
                    d1 = INFINITY
                if near_idx_map[j] == random_medoid_pos:
                    d2 = <double>second_dist[j]
                    if isnan(d2):
                        d2 = INFINITY
                    val = d2 if d2 < d else d
                    if isinf(d1):
                        if not isinf(val):
                            inf_delta -= 1
                            total_delta += val
                    elif isinf(val):
                        inf_delta += 1
                        total_delta -= d1
                    else:
                        total_delta += (val - d1)
                else:
                    if d < d1:
                        if isinf(d1):
                            inf_delta -= 1
                            total_delta += d
                        else:
                            total_delta += (d - d1)
            if inf_delta < 0:
                total_delta = -INFINITY
            elif inf_delta > 0:
                total_delta = INFINITY

    return total_delta


# FastCLARANS FastPAM1 delta calculation
def fastpam1_delta(
    const intp_t[::1] near_idx_map,
    const floating[::1] near_dist,
    const floating[::1] second_dist,
    const floating[::1] d_xc,
    Py_ssize_t n_samples,
    Py_ssize_t n_clusters,
    floating[::1] delta_buf = None,
):
    """Compute FastPAM1 delta cost for all clusters in one pass."""
    cdef:
        cnp.ndarray[floating, ndim=1] total_delta_np = None
        floating[::1] delta_arr
        double delta_td = 0.0
        Py_ssize_t j, m, best_m = 0
        double d1, d2, dc, val, best_val, tot_fin, best_fin
        double stack_buf[512]
        long long stack_inf[512]
        double* dbl_delta = NULL
        long long* inf_cnt = NULL
        long long base_inf = 0
        long long tot_inf, best_inf
        bint heap_allocated = False
        bint inf_heap_allocated = False
        bint need_inf_resolve = False
        bint inf_mem_error = False

    if delta_buf is None:
        total_delta_np = np.zeros(
            n_clusters, dtype=np.float64 if floating is double else np.float32
        )
        delta_arr = total_delta_np
    else:
        if delta_buf.shape[0] < n_clusters:
            raise ValueError(
                f"delta_buf length ({delta_buf.shape[0]}) must be >= n_clusters ({n_clusters})"
            )
        delta_arr = delta_buf

    if n_clusters <= 512:
        dbl_delta = stack_buf
    else:
        dbl_delta = <double*>malloc(n_clusters * sizeof(double))
        if dbl_delta == NULL:
            raise MemoryError(f"Failed to allocate {n_clusters} double-precision delta buffer")
        heap_allocated = True

    try:
        with nogil:
            for m in range(n_clusters):
                dbl_delta[m] = 0.0

            for j in range(n_samples):
                m = near_idx_map[j]
                d1 = <double>near_dist[j]
                d2 = <double>second_dist[j]
                dc = <double>d_xc[j]
                if isnan(dc):
                    dc = INFINITY

                if dc < d1:
                    delta_td += (dc - d1)
                elif dc < d2:
                    dbl_delta[m] += (dc - d1)
                else:
                    dbl_delta[m] += (d2 - d1)

            best_val = dbl_delta[0] + delta_td
            dbl_delta[0] = best_val

            for m in range(1, n_clusters):
                dbl_delta[m] += delta_td
                if dbl_delta[m] < best_val:
                    best_val = dbl_delta[m]
                    best_m = m

            need_inf_resolve = not isfinite(delta_td)
            if not need_inf_resolve:
                for m in range(n_clusters):
                    if not isfinite(dbl_delta[m]):
                        need_inf_resolve = True
                        break

            if need_inf_resolve:
                if n_clusters <= 512:
                    inf_cnt = stack_inf
                else:
                    inf_cnt = <long long*>malloc(n_clusters * sizeof(long long))
                    if inf_cnt == NULL:
                        inf_mem_error = True
                    else:
                        inf_heap_allocated = True

                if not inf_mem_error:
                    delta_td = 0.0
                    base_inf = 0
                    for m in range(n_clusters):
                        dbl_delta[m] = 0.0
                        inf_cnt[m] = 0

                    for j in range(n_samples):
                        m = near_idx_map[j]
                        d1 = <double>near_dist[j]
                        d2 = <double>second_dist[j]
                        dc = <double>d_xc[j]
                        if isnan(d1):
                            d1 = INFINITY
                        if isnan(d2):
                            d2 = INFINITY
                        if isnan(dc):
                            dc = INFINITY

                        if dc < d1:
                            if isinf(d1):
                                base_inf -= 1
                                delta_td += dc
                            else:
                                delta_td += (dc - d1)
                        else:
                            val = dc if dc < d2 else d2
                            if isinf(d1):
                                if not isinf(val):
                                    inf_cnt[m] -= 1
                                    dbl_delta[m] += val
                            elif isinf(val):
                                inf_cnt[m] += 1
                                dbl_delta[m] -= d1
                            else:
                                dbl_delta[m] += (val - d1)

                    best_m = 0
                    best_inf = base_inf + inf_cnt[0]
                    best_fin = delta_td + dbl_delta[0]
                    if best_inf < 0:
                        dbl_delta[0] = -INFINITY
                    elif best_inf > 0:
                        dbl_delta[0] = INFINITY
                    else:
                        dbl_delta[0] = best_fin

                    for m in range(1, n_clusters):
                        tot_inf = base_inf + inf_cnt[m]
                        tot_fin = delta_td + dbl_delta[m]
                        if tot_inf < 0:
                            dbl_delta[m] = -INFINITY
                        elif tot_inf > 0:
                            dbl_delta[m] = INFINITY
                        else:
                            dbl_delta[m] = tot_fin

                        if tot_inf < best_inf or (tot_inf == best_inf and tot_fin < best_fin):
                            best_inf = tot_inf
                            best_fin = tot_fin
                            best_m = m

                    best_val = dbl_delta[best_m]

            if not inf_mem_error:
                for m in range(n_clusters):
                    delta_arr[m] = <floating>dbl_delta[m]

        if inf_mem_error:
            raise MemoryError(f"Failed to allocate {n_clusters} int64 inf-count buffer")
    finally:
        if heap_allocated:
            free(dbl_delta)
        if inf_heap_allocated:
            free(inf_cnt)

    return best_m, best_val, total_delta_np if total_delta_np is not None else np.asarray(delta_arr)


# Nearest and second-nearest medoid distances
def update_cache_2min(
    const floating[:, ::1] subD,
    Py_ssize_t n_samples,
    Py_ssize_t n_clusters,
    object near_idx_map_buf = None,
    object near_dist_buf = None,
    object second_dist_buf = None,
    int has_nan = -1,
):
    """Find nearest and second-nearest medoid indices and distances."""
    cdef:
        object ret_idx_map
        object ret_near_dist
        object ret_second_dist

        intp_t[::1] near_idx_map
        floating[::1] near_dist
        floating[::1] second_dist

        Py_ssize_t i, m
        floating d, m1_val, m2_val
        intp_t m1_idx, m2_idx

    if near_idx_map_buf is None:
        ret_idx_map = np.empty(n_samples, dtype=np.intp)
        near_idx_map = ret_idx_map
    else:
        near_idx_map = near_idx_map_buf
        if near_idx_map.shape[0] < n_samples:
            raise ValueError(
                f"near_idx_map_buf length ({near_idx_map.shape[0]}) must be >= n_samples ({n_samples})"
            )
        ret_idx_map = near_idx_map_buf

    if near_dist_buf is None:
        ret_near_dist = np.empty(
            n_samples, dtype=np.float64 if floating is double else np.float32
        )
        near_dist = ret_near_dist
    else:
        near_dist = near_dist_buf
        if near_dist.shape[0] < n_samples:
            raise ValueError(
                f"near_dist_buf length ({near_dist.shape[0]}) must be >= n_samples ({n_samples})"
            )
        ret_near_dist = near_dist_buf

    if second_dist_buf is None:
        ret_second_dist = np.empty(
            n_samples, dtype=np.float64 if floating is double else np.float32
        )
        second_dist = ret_second_dist
    else:
        second_dist = second_dist_buf
        if second_dist.shape[0] < n_samples:
            raise ValueError(
                f"second_dist_buf length ({second_dist.shape[0]}) must be >= n_samples ({n_samples})"
            )
        ret_second_dist = second_dist_buf

    if n_clusters < 2:
        with nogil:
            for i in range(n_samples):
                near_idx_map[i] = 0
                near_dist[i] = subD[i, 0]
                second_dist[i] = INFINITY
        return ret_idx_map, ret_near_dist, ret_second_dist

    with nogil:
        # Auto-detect NaN presence if not explicitly specified (has_nan < 0)
        if has_nan < 0:
            has_nan = 0
            for i in range(n_samples):
                for m in range(n_clusters):
                    if isnan(subD[i, m]):
                        has_nan = 1
                        break
                if has_nan:
                    break

        if not has_nan:
            # Fast vectorized/branchless path: finite distances guaranteed
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
        else:
            # Robust fallback path for nan_euclidean with missing values
            for i in range(n_samples):
                m1_val = INFINITY
                m2_val = INFINITY
                m1_idx = 0
                m2_idx = 1 if n_clusters > 1 else 0

                for m in range(n_clusters):
                    d = subD[i, m]
                    if isnan(d):
                        continue
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

    return ret_idx_map, ret_near_dist, ret_second_dist


# PAM BUILD greedy selection
def pam_build_step(
    const floating[:, ::1] D,
    const intp_t[::1] candidate_indices,
    const floating[::1] dist_to_nearest,
    Py_ssize_t n_samples,
    Py_ssize_t n_candidates,
):
    """Compute distance reduction gains for candidate medoids in PAM BUILD."""
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


# k-medoids++ candidate trials
def kmedoids_pp_trials(
    const floating[::1] closest_dist_sq,
    const floating[:, ::1] dists_candidates,
    const intp_t[::1] candidate_ids,
    const intp_t[::1] current_medoids,
    Py_ssize_t n_samples,
    Py_ssize_t n_local_trials,
    Py_ssize_t n_current_medoids,
):
    """Evaluate candidate medoid potential across trials."""
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


# Distance matrix symmetry check
def is_matrix_symmetric(
    const floating[:, ::1] D,
    Py_ssize_t n_samples,
    double rtol=1e-5,
    double atol=1e-8,
):
    """Check whether a square distance matrix is symmetric within tolerance."""
    cdef:
        Py_ssize_t i, j, ii, jj, i_end, j_end
        floating d_ij, d_ji, diff, threshold
        int symmetric = 1

    with nogil:
        ii = 0
        while ii < n_samples:
            i_end = ii + 64
            if i_end > n_samples:
                i_end = n_samples

            jj = ii
            while jj < n_samples:
                j_end = jj + 64
                if j_end > n_samples:
                    j_end = n_samples

                if ii == jj:
                    for i in range(ii, i_end):
                        for j in range(i + 1, j_end):
                            d_ij = D[i, j]
                            d_ji = D[j, i]
                            if d_ij == d_ji:
                                continue
                            if isnan(d_ij) or isnan(d_ji) or isinf(d_ij) or isinf(d_ji):
                                symmetric = 0
                                break
                            diff = fabs(d_ij - d_ji)
                            threshold = atol + rtol * fabs(d_ji)
                            if diff > threshold:
                                symmetric = 0
                                break
                        if not symmetric:
                            break
                else:
                    for i in range(ii, i_end):
                        for j in range(jj, j_end):
                            d_ij = D[i, j]
                            d_ji = D[j, i]
                            if d_ij == d_ji:
                                continue
                            if isnan(d_ij) or isnan(d_ji) or isinf(d_ij) or isinf(d_ji):
                                symmetric = 0
                                break
                            diff = fabs(d_ij - d_ji)
                            threshold = atol + rtol * fabs(d_ji)
                            if diff > threshold:
                                symmetric = 0
                                break
                        if not symmetric:
                            break

                if not symmetric:
                    break
                jj += 64

            if not symmetric:
                break
            ii += 64

    return bool(symmetric)

