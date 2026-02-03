"""FastCLARANS implementation.

Provides a faster variant of CLARANS by using FastPAM1 delta-cost updates.
This implementation computes distances on-the-fly as recommended in the
original paper, making it memory-efficient for large datasets while still
benefiting from the O(k) speedup per swap evaluation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.utils.validation import check_array, check_random_state

from clarans.clarans import CLARANS
from clarans.initialization import (
    initialize_build,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


class FastCLARANS(CLARANS):
    """
    FastCLARANS: Fast variant of the CLARANS clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form (also the number of medoids).

    numlocal : int, default=2
        The number of local searches to perform. More local searches
        increase the chance of finding a better minimum at the cost of
        additional runtime.

    maxneighbor : int or None, default=None
        The maximum number of non-medoid candidates to sample per local
        search. If ``None``, defaults to 2.5% of non-medoid points
        (i.e., ``0.025 * (n - k)``) as recommended in the paper.

    max_iter : int or None, default=300
        Maximum number of successful swaps (improvements) allowed per
        local search. Use ``None`` to disable this safeguard.

    init : {'random', 'heuristic', 'k-medoids++', 'build', array-like}, default='random'
        Method for initialization. If an array-like is provided it should
        be of shape (n_clusters, n_features) and will be snapped to the
        nearest points in X.

    metric : str or callable, default='euclidean'
        The distance metric passed to scikit-learn pairwise utilities.

    random_state : int, RandomState instance or None, default=None
        Controls random number generation for reproducibility.

    Attributes
    ----------
    medoid_indices_ : ndarray of shape (n_clusters,)
        Indices of the selected medoids in the training set.

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of the medoids (rows from the training data).

    labels_ : ndarray of shape (n_samples,)
        Labels of each point indicating the nearest medoid.

    Notes
    -----
    This implementation follows the original FastCLARANS paper by computing
    distances on-the-fly rather than precomputing a full distance matrix.
    This keeps memory usage at O(n) instead of O(n^2), making it suitable
    for larger datasets.
    
    The key improvement from FastCLARANS is the sampling strategy: instead
    of sampling random (medoid, non-medoid) pairs like CLARANS, it samples
    only non-medoid candidates and evaluates swaps with all k medoids at
    once using FastPAM1 delta formulas. This explores k edges of the search
    graph in the time CLARANS explores one.

    References
    ----------
    Schubert, E., & Rousseeuw, P. J. (2021). Fast and eager k-medoids
    clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS
    algorithms. Information Systems, 101, 101804.
    """

    def fit(self, X: ArrayLike | "spmatrix", y: Any = None) -> "FastCLARANS":
        """
        Fit the FastCLARANS model to X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training instances to cluster. Accepts CSR/CSC sparse matrices.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : FastCLARANS
            The fitted estimator. Attributes set on the estimator include
            ``medoid_indices_``, ``cluster_centers_`` and ``labels_``.

        Raises
        ------
        ValueError
            If ``n_clusters >= n_samples`` or if an explicit ``init`` array
            is provided with an incompatible shape or there are not enough
            unique points to form the requested number of medoids.

        Notes
        -----
        Unlike implementations that precompute the full distance matrix,
        this version computes distances on-the-fly to save memory. This
        is efficient for low-dimensional data with cheap distance metrics
        (e.g., Euclidean distance).
        """
        X = check_array(X, accept_sparse=["csr", "csc"])
        self.n_features_in_ = X.shape[1]

        random_state = check_random_state(self.random_state)
        n_samples, _ = X.shape

        if self.n_clusters >= n_samples:
            raise ValueError("n_clusters must be less than n_samples")

        if self.maxneighbor is None:
            # FastCLARANS samples 2.5% of non-medoid points per local search
            # (Schubert & Rousseeuw, 2021) instead of 1.25% * k * (n-k) edges
            self.maxneighbor_ = max(
                250, int(0.025 * (n_samples - self.n_clusters))
            )
        else:
            self.maxneighbor_ = self.maxneighbor

        best_cost = np.inf
        best_medoids = None
        self.n_iter_ = 0

        for loc_idx in range(self.numlocal):
            if isinstance(self.init, str) and self.init == "random":
                current_medoids_indices = random_state.choice(
                    n_samples, self.n_clusters, replace=False
                )
            elif isinstance(self.init, str) and self.init == "k-medoids++":
                current_medoids_indices = initialize_k_medoids_plus_plus(
                    X, self.n_clusters, random_state, self.metric
                )
            elif isinstance(self.init, str) and self.init == "heuristic":
                current_medoids_indices = initialize_heuristic(
                    X, self.n_clusters, self.metric
                )
            elif isinstance(self.init, str) and self.init == "build":
                current_medoids_indices = initialize_build(
                    X, self.n_clusters, self.metric
                )
            elif hasattr(self.init, "__array__") or isinstance(self.init, list):
                init_centers = check_array(self.init)
                if init_centers.shape != (self.n_clusters, self.n_features_in_):
                    raise ValueError(
                        f"init array must be of shape ({self.n_clusters}, {self.n_features_in_})"
                    )

                current_medoids_indices = pairwise_distances_argmin_min(
                    init_centers, X, metric=self.metric
                )[0]

                if len(set(current_medoids_indices)) < self.n_clusters:
                    warnings.warn(
                        "Provided init centers map to duplicate points in X. "
                        "Filling duplicates with random points."
                    )
                    current_medoids_indices = list(set(current_medoids_indices))
                    remaining = self.n_clusters - len(current_medoids_indices)
                    available = list(
                        set(range(n_samples)) - set(current_medoids_indices)
                    )
                    if len(available) < remaining:
                        raise ValueError(
                            "Not enough unique points to fill up to n_clusters."
                        )
                    current_medoids_indices.extend(
                        random_state.choice(available, remaining, replace=False)
                    )
                    current_medoids_indices = np.array(current_medoids_indices)
            else:
                raise ValueError(f"Unknown init method: {self.init}")

            current_medoids_indices.sort()

            # Compute nearest/second-nearest on-the-fly (no precomputed matrix)
            near_idx_map, near_dist, second_dist = self._update_cache_onthefly(
                X, current_medoids_indices
            )
            current_cost = np.sum(near_dist)

            i = 0
            iter_count = 0

            while i < self.maxneighbor_:
                if self.max_iter is not None and iter_count >= self.max_iter:
                    break

                # Choose a random non-medoid candidate
                while True:
                    candidate_idx = random_state.randint(0, n_samples)
                    if candidate_idx not in current_medoids_indices:
                        break

                # Compute distances from candidate to all points on-the-fly
                d_xc = pairwise_distances(
                    X[candidate_idx].reshape(1, -1), X, metric=self.metric
                ).ravel()

                removal_loss = np.zeros(self.n_clusters)
                diff = second_dist - near_dist
                removal_loss += np.bincount(
                    near_idx_map, weights=diff, minlength=self.n_clusters
                )

                mask_better_than_nearest = d_xc < near_dist
                delta_td_plus_xc = np.sum(
                    d_xc[mask_better_than_nearest] - near_dist[mask_better_than_nearest]
                )

                total_delta = removal_loss + delta_td_plus_xc

                mask_better_than_second = d_xc < second_dist

                term1 = (
                    near_dist[mask_better_than_nearest]
                    - second_dist[mask_better_than_nearest]
                )
                idx1 = near_idx_map[mask_better_than_nearest]
                total_delta += np.bincount(
                    idx1, weights=term1, minlength=self.n_clusters
                )

                mask_case2 = (~mask_better_than_nearest) & mask_better_than_second
                term2 = d_xc[mask_case2] - second_dist[mask_case2]
                idx2 = near_idx_map[mask_case2]
                total_delta += np.bincount(
                    idx2, weights=term2, minlength=self.n_clusters
                )

                min_delta_idx = np.argmin(total_delta)
                min_delta = total_delta[min_delta_idx]

                if min_delta < 0:
                    current_medoids_indices[min_delta_idx] = candidate_idx
                    current_medoids_indices.sort()

                    current_cost += min_delta

                    # Update nearest/second caches after an accepted swap
                    near_idx_map, near_dist, second_dist = self._update_cache_onthefly(
                        X, current_medoids_indices
                    )

                    i = 0
                    iter_count += 1
                else:
                    i += 1

            self.n_iter_ += max(1, iter_count)

            if current_cost < best_cost:
                best_cost = current_cost
                best_medoids = current_medoids_indices.copy()

        self.medoid_indices_ = best_medoids
        self.cluster_centers_ = X[self.medoid_indices_]

        self.labels_, _ = pairwise_distances_argmin_min(
            X, self.cluster_centers_, metric=self.metric
        )

        return self

    def _update_cache_onthefly(
        self, X: ArrayLike, medoids_indices: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute nearest and second-nearest medoid information on-the-fly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        medoids_indices : array-like of shape (n_clusters,)
            Indices of the current medoids.

        Returns
        -------
        near_idx_map : ndarray of shape (n_samples,)
            For each sample, the index (0..k-1) of the nearest medoid in
            ``medoids_indices``.

        near_dist : ndarray of shape (n_samples,)
            Distance from each sample to its nearest medoid.

        second_dist : ndarray of shape (n_samples,)
            Distance from each sample to its second nearest medoid. If
            ``n_clusters == 1`` this will be an array filled with
            ``np.inf``.
        """
        n_samples = X.shape[0]
        medoids = X[medoids_indices]
        
        # Compute distances from all points to all medoids on-the-fly
        subD = pairwise_distances(X, medoids, metric=self.metric)

        if self.n_clusters >= 2:
            partitioned_idx = np.argpartition(subD, 1, axis=1)
            smallest_idx = partitioned_idx[:, 0]
            second_smallest_idx = partitioned_idx[:, 1]

            near_dist = subD[np.arange(n_samples), smallest_idx]
            second_dist = subD[np.arange(n_samples), second_smallest_idx]
            near_idx_map = smallest_idx
        else:
            near_dist = subD[:, 0]
            second_dist = np.full(n_samples, np.inf)
            near_idx_map = np.zeros(n_samples, dtype=int)

        return near_idx_map, near_dist, second_dist
