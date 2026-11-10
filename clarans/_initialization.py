"""
Initialization strategies for k-medoids clustering.

The initialization methods ('k-medoids++', 'heuristic', 'build') are
adapted from the `scikit-learn-extra` KMedoids implementation:
https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_array, check_random_state

from .utils import _core, _pairwise_fast, _to_dense, check_medoids


def _check_init_args(X: Any, n_clusters: int) -> int:
    """Validate dataset shape and cluster count for initialization."""
    n_samples = X.shape[0]
    if n_clusters >= n_samples:
        raise ValueError(f"n_clusters must be less than n_samples ({n_samples}); got {n_clusters}")
    return n_samples


def initialize_heuristic(
    X: Any,
    n_clusters: int,
    metric: str | Callable = "euclidean",
    metric_params: dict[str, Any] | None = None,
) -> np.ndarray:
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
    _check_init_args(X, n_clusters)

    if metric == "precomputed":
        D = X
    else:
        D = _pairwise_fast(X, metric=metric, metric_params=metric_params)

    dist_sums = np.asarray(D.sum(axis=0)).ravel()
    current_medoids_indices = np.argpartition(dist_sums, n_clusters - 1)[:n_clusters]
    current_medoids_indices.sort()  # Keep indices sorted for consistency
    return current_medoids_indices


def initialize_build(
    X: Any,
    n_clusters: int,
    metric: str | Callable = "euclidean",
    metric_params: dict[str, Any] | None = None,
) -> np.ndarray:
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
    n_samples = _check_init_args(X, n_clusters)

    medoid_indices = np.empty(n_clusters, dtype=int)

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
        D = _to_dense(X)
    else:
        D = _pairwise_fast(X, metric=metric, metric_params=metric_params)

    dist_sums = np.asarray(D.sum(axis=0)).ravel()
    first_medoid = int(np.argmin(dist_sums))
    medoid_indices[0] = first_medoid

    use_cython_build = (
        _core is not None
        and isinstance(D, np.ndarray)
        and D.flags.c_contiguous
        and D.dtype in (np.float64, np.float32)
    )

    is_medoid = np.zeros(n_samples, dtype=bool)
    is_medoid[first_medoid] = True
    dist_to_nearest = (
        np.ascontiguousarray(D[:, first_medoid], dtype=D.dtype)
        if use_cython_build
        else D[:, first_medoid]
    )

    for c in range(1, n_clusters):
        candidate_indices = np.flatnonzero(~is_medoid)
        if use_cython_build:
            cand_c = np.ascontiguousarray(candidate_indices, dtype=np.intp)
            best_idx_in_cand, _ = _core.pam_build_step(
                D, cand_c, dist_to_nearest, n_samples, len(cand_c)
            )
            best_candidate = int(candidate_indices[best_idx_in_cand])
        else:
            D_candidates = D[:, candidate_indices]
            diffs = dist_to_nearest[:, np.newaxis] - D_candidates
            gains = np.sum(np.maximum(diffs, 0), axis=0)
            best_candidate = int(candidate_indices[np.argmax(gains)])

        medoid_indices[c] = best_candidate
        is_medoid[best_candidate] = True
        dist_to_nearest = np.minimum(dist_to_nearest, D[:, best_candidate])

    return medoid_indices


def _sample_distinct_candidate(
    rng: np.random.RandomState,
    n_samples: int,
    current_medoids: np.ndarray,
) -> int:
    """Sample a random sample index not in current_medoids in O(1) expected memory and time."""
    medoids_set = set(current_medoids)
    for _ in range(100):
        cand = int(rng.randint(0, n_samples))
        if cand not in medoids_set:
            return cand
    remaining = np.setdiff1d(np.arange(n_samples), current_medoids)
    return int(rng.choice(remaining))


def initialize_k_medoids_plus_plus(
    X: Any,
    n_clusters: int,
    random_state: int | np.random.RandomState | None = None,
    metric: str | Callable = "euclidean",
    n_local_trials: int | None = None,
    metric_params: dict[str, Any] | None = None,
) -> np.ndarray:
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
    rng = check_random_state(random_state)
    n_samples = _check_init_args(X, n_clusters)
    medoid_indices = np.empty(n_clusters, dtype=int)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    first_medoid = rng.randint(0, n_samples)
    medoid_indices[0] = first_medoid

    if metric == "precomputed":
        closest = _to_dense(X[:, first_medoid]).ravel()
    else:
        first_row = X[first_medoid : first_medoid + 1]
        closest = _pairwise_fast(
            X,
            first_row,
            metric=metric,
            metric_params=metric_params,
        ).flatten()

    closest_dist_sq = closest**2
    current_pot = float(closest_dist_sq.sum())

    for c in range(1, n_clusters):
        if current_pot <= 1e-16:
            medoid_indices[c] = _sample_distinct_candidate(rng, n_samples, medoid_indices[:c])
            continue

        rand_vals = rng.random_sample(n_local_trials) * current_pot

        cumsum_dist = np.cumsum(closest_dist_sq)

        candidate_ids = np.searchsorted(cumsum_dist, rand_vals)
        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        if metric == "precomputed":
            cols_dense = _to_dense(X[:, candidate_ids])
            dists_candidates = np.ascontiguousarray((cols_dense.T) ** 2)
        else:
            candidates_X = X[candidate_ids]
            cand_dists = _pairwise_fast(X, candidates_X, metric=metric, metric_params=metric_params)
            dists_candidates = np.ascontiguousarray((cand_dists.T) ** 2)

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
            best_candidate = _sample_distinct_candidate(rng, n_samples, medoid_indices[:c])
            if metric == "precomputed":
                row_dist = _to_dense(X[:, best_candidate]).ravel()
            else:
                cand_row = X[best_candidate : best_candidate + 1]
                row_dist = _pairwise_fast(
                    X, cand_row, metric=metric, metric_params=metric_params
                ).ravel()
            best_dist_sq = np.minimum(closest_dist_sq, row_dist**2)
            best_pot = float(best_dist_sq.sum())

        assert best_pot is not None and best_dist_sq is not None
        medoid_indices[c] = best_candidate
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return medoid_indices


def initialize_from_array(
    init: Any,
    X: Any,
    random_state: np.random.RandomState,
    n_clusters: int,
    metric: str | Callable = "euclidean",
    metric_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Map user-provided initial medoids or centers array to dataset observation indices.

    Snaps coordinates to the nearest unique training observations in X and automatically
    fills duplicate assignments with randomly chosen available training points.

    Parameters
    ----------
    init : array-like of shape (n_clusters, n_features) or (n_clusters,)
        The user-provided coordinates or medoid indices.
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The validated input data matrix.
    random_state : RandomState
        Random number generator.
    n_clusters : int
        Number of clusters to form.
    metric : str or callable, default='euclidean'
        Distance metric.
    metric_params : dict, default=None
        Additional metric keyword parameters.

    Returns
    -------
    current_medoids_indices : ndarray of shape (n_clusters,), dtype int
        Indices of the selected medoids in the dataset.
    """
    n_samples, n_features = X.shape

    init_arr = np.asarray(init)

    if metric == "precomputed":
        if (
            not issparse(init)
            and init_arr.ndim == 1
            and len(init_arr) == n_clusters
            and np.issubdtype(init_arr.dtype, np.integer)
        ):
            current_medoids_indices = check_medoids(init_arr, n_samples=n_samples)
            return current_medoids_indices

        init_centers = check_array(init, accept_sparse=["csr", "csc"])
        if init_centers.shape != (n_clusters, n_features):
            raise ValueError(f"init array must be of shape ({n_clusters}, {n_features})")
        current_medoids_indices = np.asarray(init_centers.argmin(axis=1)).ravel()
    else:
        try:
            init_centers = check_array(init, accept_sparse=["csr", "csc"])
        except ValueError as err:
            if "Expected 2D array, got 1D array instead" in str(err):
                raise ValueError(
                    f"init array must be 2D of shape ({n_clusters}, {n_features}). "
                    "If you want to specify medoid indices, use metric='precomputed' "
                    "or pass X[indices] as initial centers."
                ) from err
            raise
        if init_centers.shape != (n_clusters, n_features):
            raise ValueError(f"init array must be of shape ({n_clusters}, {n_features})")

        params = metric_params if metric_params is not None else {}
        D = _pairwise_fast(init_centers, X, metric=metric, metric_params=params)
        current_medoids_indices = np.argmin(D, axis=1)

    current_medoids_indices = np.unique(current_medoids_indices)

    if len(current_medoids_indices) < n_clusters:
        warnings.warn(
            "Provided init centers map to duplicate points in X. "
            "Filling duplicates with random points.",
            UserWarning,
            stacklevel=2,
        )
        remaining = n_clusters - len(current_medoids_indices)
        all_indices = np.arange(n_samples)
        available = np.setdiff1d(all_indices, current_medoids_indices, assume_unique=True)

        if len(available) < remaining:
            raise ValueError("Not enough unique points to fill up to n_clusters.")

        fillers = random_state.choice(available, remaining, replace=False)
        current_medoids_indices = np.concatenate([current_medoids_indices, fillers])

    return current_medoids_indices
