"""
Initialization strategies for k-medoids clustering.

The initialization methods ('k-medoids++', 'heuristic', 'build') are
adapted from the `scikit-learn-extra` KMedoids implementation:
https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
"""

import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from sklearn.metrics import DistanceMetric, pairwise_distances
from sklearn.utils import check_random_state

from .utils import _SCIPY_METRIC_MAP


def _compute_pairwise_distances(X, Y=None, metric="euclidean", metric_params=None):
    """Compute pairwise distances using cdist for dense arrays when possible,
    falling back to scikit-learn's DistanceMetric and pairwise_distances."""
    scipy_metric = _SCIPY_METRIC_MAP.get(metric, metric) if isinstance(metric, str) else metric
    params = metric_params if metric_params is not None else {}

    if not issparse(X) and (Y is None or not issparse(Y)):
        try:
            if Y is None:
                return cdist(X, X, metric=scipy_metric, **params)
            return cdist(X, Y, metric=scipy_metric, **params)
        except Exception:
            pass
    else:
        # DistanceMetric directly supports sparse matrices for metrics like
        # chebyshev, canberra, cityblock, etc.
        if isinstance(scipy_metric, str):
            try:
                dm = DistanceMetric.get_metric(scipy_metric, **params)
                if Y is None:
                    return dm.pairwise(X)
                return dm.pairwise(X, Y)
            except Exception:
                pass

    if Y is None:
        return pairwise_distances(X, metric=scipy_metric, **params)
    return pairwise_distances(X, Y, metric=scipy_metric, **params)


try:
    from . import _core
except ImportError:
    _core = None  # type: ignore[assignment]


def initialize_heuristic(X, n_clusters, metric="euclidean", metric_params=None):
    """
    Initialize medoids using a heuristic approach.

    Picks the n_clusters points with the smallest sum distance to every other point.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    n_clusters : int
        The number of clusters to form.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    Returns
    -------
    current_medoids_indices : ndarray of shape (n_clusters,), dtype int
        Indices of the selected medoids in the dataset.

    Notes
    -----
    This method computes the full pairwise distance matrix and therefore has
    O(n^2) time and memory complexity.

    References
    ----------
    Adapted from the scikit-learn-extra KMedoids implementation:
    https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    """
    n_samples = X.shape[0]
    if n_clusters >= n_samples:
        raise ValueError(
            f"n_clusters must be less than n_samples ({n_samples}); got {n_clusters}"
        )

    if metric == "precomputed":
        D = X
    else:
        D = _compute_pairwise_distances(X, metric=metric, metric_params=metric_params)

    if hasattr(D, "toarray"):
        dist_sums = np.asarray(D.sum(axis=1)).ravel()
    else:
        dist_sums = np.sum(D, axis=1)
    current_medoids_indices = np.argpartition(dist_sums, n_clusters - 1)[:n_clusters]
    return current_medoids_indices


def initialize_build(X, n_clusters, metric="euclidean", metric_params=None):
    """
    Initialize medoids using the PAM BUILD step.

    Greedily selects the first medoid that minimizes total distance, then
    subsequently adds medoids that maximally decrease the total cost.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    n_clusters : int
        The number of clusters to form.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    Returns
    -------
    current_medoids_indices : ndarray of shape (n_clusters,), dtype int
        Indices of the selected medoids in the dataset.

    Notes
    -----
    This method computes the full pairwise distance matrix and therefore has
    O(n^2) time and memory complexity.
    This method implements the greedy BUILD phase from PAM (Kaufman & Rousseeuw, 1990).

    References
    ----------
    Adapted from the scikit-learn-extra KMedoids implementation:
    https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    """
    n_samples = X.shape[0]
    if n_clusters >= n_samples:
        raise ValueError(
            f"n_clusters must be less than n_samples ({n_samples}); got {n_clusters}"
        )

    medoids = []

    if metric == "precomputed":
        if hasattr(X, "toarray"):
            warnings.warn(
                "The 'build' initialization does not support sparse distance matrices directly "
                "and will convert the matrix to a dense array via `.toarray()`. "
                "This may consume significant memory. "
                "Consider using init='heuristic' or init='k-medoids++' "
                "to preserve memory efficiency.",
                UserWarning,
                stacklevel=3,
            )
            D = X.toarray()
        else:
            D = np.asarray(X)
    else:
        D = _compute_pairwise_distances(X, metric=metric, metric_params=metric_params)

    dist_sums = D.sum(axis=1)
    first_medoid = int(np.argmin(dist_sums))
    medoids.append(first_medoid)

    dist_to_nearest = D[:, first_medoid]

    use_cython_build = (
        _core is not None
        and isinstance(D, np.ndarray)
        and D.flags.c_contiguous
        and D.dtype in (np.float64, np.float32)
    )

    if use_cython_build:
        dist_to_nearest_c = np.ascontiguousarray(dist_to_nearest, dtype=D.dtype)
        is_medoid = np.zeros(n_samples, dtype=bool)
        is_medoid[first_medoid] = True

        for _ in range(1, n_clusters):
            candidate_indices = np.ascontiguousarray(np.where(~is_medoid)[0], dtype=np.intp)
            best_idx_in_cand, _ = _core.pam_build_step(
                D, candidate_indices, dist_to_nearest_c, n_samples, len(candidate_indices)
            )
            best_candidate = int(candidate_indices[best_idx_in_cand])
            medoids.append(best_candidate)
            is_medoid[best_candidate] = True
            dist_to_nearest_c = np.minimum(dist_to_nearest_c, D[:, best_candidate])
    else:
        for _ in range(1, n_clusters):
            # Mask for medoids
            is_medoid = np.zeros(n_samples, dtype=bool)
            is_medoid[medoids] = True

            # We only care about candidates
            candidate_indices = np.where(~is_medoid)[0]

            D_candidates = D[:, candidate_indices]
            diffs = dist_to_nearest[:, np.newaxis] - D_candidates
            gains = np.sum(np.maximum(diffs, 0), axis=0)

            best_candidate_idx_in_candidates = np.argmax(gains)
            best_candidate = int(candidate_indices[best_candidate_idx_in_candidates])

            medoids.append(best_candidate)

            dist_to_nearest = np.minimum(dist_to_nearest, D[:, best_candidate])

    return np.array(medoids, dtype=int)


def initialize_k_medoids_plus_plus(
    X, n_clusters, random_state=None, metric="euclidean", n_local_trials=None, metric_params=None
):
    """
    Initialize medoids using k-medoids++ (similar to k-means++).

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    n_clusters : int
        The number of clusters to form.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for initial medoid selection.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.

    n_local_trials : int, default=None
        The number of local seeding trials for each center. If None,
        defaults to ``2 + int(np.log(n_clusters))`` as recommended by Arthur & Vassilvitskii.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    Returns
    -------
    medoid_indices : ndarray of shape (n_clusters,), dtype int
        Indices of the selected medoids in the dataset.

    Notes
    -----
    This implementation follows the k-means++ style seeding (Arthur & Vassilvitskii, 2007)
    but uses distances squared and picks medoids (data indices) rather than centroids.

    References
    ----------
    Adapted from the scikit-learn-extra KMedoids implementation:
    https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]
    if n_clusters >= n_samples:
        raise ValueError(
            f"n_clusters must be less than n_samples ({n_samples}); got {n_clusters}"
        )
    medoid_indices = np.empty(n_clusters, dtype=int)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    first_medoid = random_state.randint(0, n_samples)
    medoid_indices[0] = first_medoid

    first_row = X[first_medoid : first_medoid + 1]
    if metric == "precomputed":
        closest = (
            first_row.toarray().ravel()
            if hasattr(first_row, "toarray")
            else np.asarray(first_row).ravel()
        )
    else:
        closest = _compute_pairwise_distances(
            X,
            first_row,
            metric=metric,
            metric_params=metric_params,
        ).flatten()

    closest_dist_sq = closest**2
    current_pot = float(closest_dist_sq.sum())

    for c in range(1, n_clusters):
        if current_pot <= 1e-16:
            remaining = np.setdiff1d(np.arange(n_samples), medoid_indices[:c])
            chosen_candidate = int(random_state.choice(remaining))
            medoid_indices[c] = chosen_candidate
            continue

        rand_vals = random_state.random_sample(n_local_trials) * current_pot

        cumsum_dist = np.cumsum(closest_dist_sq)

        candidate_ids = np.searchsorted(cumsum_dist, rand_vals)
        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        if metric == "precomputed":
            dists_candidates = (
                (
                    X[candidate_ids].toarray()
                    if hasattr(X, "toarray")
                    else np.asarray(X[candidate_ids])
                )
                ** 2
            )
        else:
            candidates_X = X[candidate_ids]
            dists_candidates = (
                _compute_pairwise_distances(
                    candidates_X, X, metric=metric, metric_params=metric_params
                )
                ** 2
            )

        best_candidate = None
        best_pot = None
        best_dist_sq = None

        use_cython_pp = (
            _core is not None
            and isinstance(dists_candidates, np.ndarray)
            and dists_candidates.flags.c_contiguous
            and dists_candidates.dtype in (np.float64, np.float32)
            and isinstance(closest_dist_sq, np.ndarray)
            and closest_dist_sq.flags.c_contiguous
            and closest_dist_sq.dtype == dists_candidates.dtype
        )

        if use_cython_pp:
            candidate_ids_c = np.ascontiguousarray(candidate_ids, dtype=np.intp)
            current_medoids_c = np.ascontiguousarray(medoid_indices[:c], dtype=np.intp)
            cand_res, pot_res, dist_sq_res = _core.kmedoids_pp_trials(
                closest_dist_sq,
                dists_candidates,
                candidate_ids_c,
                current_medoids_c,
                n_samples,
                n_local_trials,
                c,
            )
            if cand_res >= 0:
                best_candidate = int(cand_res)
                best_pot = float(pot_res)
                best_dist_sq = dist_sq_res

        if best_candidate is None:
            for i in range(n_local_trials):
                cand_id = int(candidate_ids[i])
                if cand_id in medoid_indices[:c]:
                    continue
                new_dist_sq = np.minimum(closest_dist_sq, dists_candidates[i])
                new_pot = new_dist_sq.sum()

                if best_candidate is None or new_pot < best_pot:
                    best_candidate = cand_id
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

        if best_candidate is None:
            remaining = np.setdiff1d(np.arange(n_samples), medoid_indices[:c])
            best_candidate = int(random_state.choice(remaining))
            cand_row = X[best_candidate : best_candidate + 1]
            if metric == "precomputed":
                row_dist = (
                    cand_row.toarray().ravel()
                    if hasattr(cand_row, "toarray")
                    else np.asarray(cand_row).ravel()
                )
            else:
                row_dist = _compute_pairwise_distances(cand_row, X, metric=metric).ravel()
            best_dist_sq = np.minimum(closest_dist_sq, row_dist**2)
            best_pot = float(best_dist_sq.sum())

        medoid_indices[c] = best_candidate
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return medoid_indices
