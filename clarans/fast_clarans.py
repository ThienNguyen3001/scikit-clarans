"""FastCLARANS implementation.

Provides a faster variant of CLARANS by using FastPAM1 delta-cost updates.
This implementation computes distances on-the-fly as recommended in the
original paper, making it memory-efficient for large datasets while still
benefiting from the O(k) speedup per swap evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_distances

from clarans.clarans import CLARANS, _DELTA_TOL

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


class FastCLARANS(CLARANS):
    """
    FastCLARANS: Fast variant of the CLARANS clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form (also the number of medoids).

    num_local : int, default=2
        The number of local searches to perform. More local searches
        increase the chance of finding a better minimum at the cost of
        additional runtime.

    max_neighbors : int or 'auto', default='auto'
        The maximum number of non-medoid candidates to sample per local
        search. If ``'auto'``, defaults to 2.5% of non-medoid points
        (i.e., ``0.025 * (n - k)``) as recommended in Schubert & Rousseeuw
        (2021). This adaptive default automatically scales with dataset size
        without requiring manual tuning.

    init : {'k-medoids++', 'random', 'heuristic', 'build', array-like}, default='k-medoids++'
        Method for initialization. If an array-like is provided it should
        be of shape (n_clusters, n_features) and will be snapped to the
        nearest points in X.

    metric : str or callable, default='euclidean'
        The distance metric passed to scikit-learn pairwise utilities.

    random_state : int, RandomState instance or None, default=None
        Controls random number generation for reproducibility.

    Attributes
    ----------
    cluster_centers_ : {ndarray, sparse matrix} of shape (n_clusters, n_features) or None
        Coordinates of cluster centers (medoids). If ``metric='precomputed'``,
        this is ``None``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point indicating the nearest medoid.

    medoid_indices_ : ndarray of shape (n_clusters,)
        Indices of the selected medoids in the training set.

    inertia_ : float
        Sum of distances from each sample to its nearest medoid (total
        cost of the best solution found).

    max_neighbors_ : int
        Effective maximum number of non-improving non-medoid candidates
        sampled per local search.

    n_iter_ : int
        Number of candidate neighbors evaluated during the best local search.

    n_swaps_ : int
        Number of successful medoid swaps performed during the best local
        search.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Defined only when
        ``metric != 'precomputed'``.

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

        y : Ignored, default=None
            Not used, present for API consistency with scikit-learn
            pipelines and ClusterMixin.

        Returns
        -------
        self : FastCLARANS
            The fitted estimator. Attributes set on the estimator include
            ``medoid_indices_``, ``cluster_centers_``, ``labels_`` and
            ``inertia_``.

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
        X, random_state, n_samples, n_features = self._validate_input_and_params(X)

        if self.max_neighbors == "auto":
            # FastCLARANS samples 2.5% of non-medoid points per local search
            # (Schubert & Rousseeuw, 2021) instead of 1.25% * k * (n-k) edges
            self.max_neighbors_ = max(
                250, int(0.025 * (n_samples - self.n_clusters))
            )
        else:
            self.max_neighbors_ = int(self.max_neighbors)

        best_cost = np.inf
        best_medoids: np.ndarray = np.empty(self.n_clusters, dtype=int)
        best_n_iter = 0
        best_n_swaps = 0

        deterministic_medoids = self._prepare_initial_medoids(X, random_state)

        for loc_idx in range(self.num_local):
            if deterministic_medoids is not None:
                current_medoids_indices = deterministic_medoids.copy()
            else:
                current_medoids_indices = self._initialize_medoids(X, random_state)
            current_medoids_indices.sort()
            
            near_idx_map, near_dist, second_dist = self._update_cache(
                X, current_medoids_indices
            )
            current_cost: float = float(np.sum(near_dist))

            i = 0
            swap_count = 0
            eval_count = 0

            while i < self.max_neighbors_:
                eval_count += 1
                # Choose a random non-medoid candidate using mask (safe for
                # any k/n ratio, avoids rejection sampling infinite loop)
                non_medoid_mask = np.ones(n_samples, dtype=bool)
                non_medoid_mask[current_medoids_indices] = False
                available_candidates = np.flatnonzero(non_medoid_mask)

                if available_candidates.size == 0:
                    break

                candidate_idx = random_state.choice(available_candidates)

                # Compute distances from candidate to all points
                cand_row = X[candidate_idx : candidate_idx + 1]
                if self.metric == "precomputed":
                    d_xc = (
                        cand_row.toarray().ravel()
                        if hasattr(cand_row, "toarray")
                        else np.asarray(cand_row).ravel()
                    )
                else:
                    d_xc = pairwise_distances(
                        cand_row, X, metric=self.metric
                    ).ravel()

                if self.n_clusters == 1:
                    candidate_cost = float(np.sum(d_xc))
                    min_delta = candidate_cost - current_cost
                    min_delta_idx = 0
                else:
                    removal_loss = np.zeros(self.n_clusters)
                    diff = second_dist - near_dist
                    with np.errstate(invalid="ignore"):
                        removal_loss += np.bincount(
                            near_idx_map, weights=diff, minlength=self.n_clusters
                        )

                    mask_better_than_nearest = d_xc < near_dist
                    delta_td_plus_xc: float = float(
                        np.sum(
                            d_xc[mask_better_than_nearest]
                            - near_dist[mask_better_than_nearest]
                        )
                    )

                    total_delta = removal_loss + delta_td_plus_xc

                    mask_better_than_second = d_xc < second_dist

                    term1 = (
                        near_dist[mask_better_than_nearest]
                        - second_dist[mask_better_than_nearest]
                    )
                    idx1 = near_idx_map[mask_better_than_nearest]
                    with np.errstate(invalid="ignore"):
                        total_delta += np.bincount(
                            idx1, weights=term1, minlength=self.n_clusters
                        )

                    mask_case2 = (~mask_better_than_nearest) & mask_better_than_second
                    term2 = d_xc[mask_case2] - second_dist[mask_case2]
                    idx2 = near_idx_map[mask_case2]
                    with np.errstate(invalid="ignore"):
                        total_delta += np.bincount(
                            idx2, weights=term2, minlength=self.n_clusters
                        )

                    min_delta_idx = int(np.argmin(total_delta))
                    min_delta = total_delta[min_delta_idx]

                if min_delta < _DELTA_TOL:
                    current_medoids_indices[min_delta_idx] = candidate_idx
                    current_medoids_indices.sort()

                    # Update nearest/second caches after an accepted swap
                    near_idx_map, near_dist, second_dist = self._update_cache(
                        X, current_medoids_indices
                    )
                    current_cost = float(np.sum(near_dist))

                    i = 0
                    swap_count += 1
                else:
                    i += 1

            if current_cost < best_cost + _DELTA_TOL:
                best_cost = current_cost
                best_medoids = current_medoids_indices.copy()
                best_n_iter = eval_count
                best_n_swaps = swap_count

        self.n_iter_ = best_n_iter
        self.n_swaps_ = best_n_swaps

        return self._finalize_fit(X, best_cost, best_medoids)

    def _update_cache(
        self,
        X: np.ndarray | spmatrix,
        medoids_indices: Sequence[int] | np.ndarray,
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
        if self.metric == "precomputed":
            sub_mat = X[:, medoids_indices]
            subD = (
                sub_mat.toarray()
                if hasattr(sub_mat, "toarray")
                else np.asarray(sub_mat)
            )
        else:
            medoids = X[medoids_indices]
            subD = pairwise_distances(X, medoids, metric=self.metric)

        if self.n_clusters >= 2:
            # Use argsort on the (n_samples, k) matrix to correctly
            # identify the nearest and second-nearest medoids.
            # np.argpartition(subD, 1) does NOT guarantee that index 0
            # holds the smallest value — only that the element at
            # position 1 is the correct partition boundary.  argsort
            # on a small k-column matrix is cheap and avoids this bug.
            sorted_idx = np.argsort(subD, axis=1)
            smallest_idx = sorted_idx[:, 0]
            second_smallest_idx = sorted_idx[:, 1]

            near_dist = subD[np.arange(n_samples), smallest_idx]
            second_dist = subD[np.arange(n_samples), second_smallest_idx]
            near_idx_map = smallest_idx
        else:
            near_dist = subD[:, 0]
            second_dist = np.full(n_samples, np.inf)
            near_idx_map = np.zeros(n_samples, dtype=int)

        return near_idx_map, near_dist, second_dist
