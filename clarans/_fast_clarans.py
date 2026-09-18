"""FastCLARANS implementation.

Provides a faster variant of CLARANS by using FastPAM1 delta-cost updates.
This implementation computes distances on-the-fly as recommended in the
original paper, making it memory-efficient for large datasets while still
benefiting from the O(k) speedup per swap evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ._clarans import CLARANS, _DELTA_TOL
from .utils import _warn_cython_unavailable

try:
    from . import _core
except ImportError:
    _core = None  # type: ignore[assignment]

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
        search. If ``'auto'``, defaults to ``max(1, int(250 / k), int(0.025 * (n - k)))``
        combining the 2.5% non-medoid sample recommended by Schubert & Rousseeuw
        (2021) with a proportional floor equivalent to 250 edge evaluations
        (Ng & Han, 2002). This adaptive default automatically scales with dataset
        size without requiring manual tuning.


        .. note::
            **Node vs. Edge Sampling & Parameter Comparison**:
            In classic CLARANS, each neighbor step evaluates a single pair
            ``(medoid, candidate)`` (1 graph edge). In FastCLARANS, each neighbor
            step samples 1 non-medoid candidate and evaluates swaps against
            **all k medoids simultaneously** via FastPAM1 delta caching.
            Thus, 1 neighbor step in FastCLARANS evaluates ``k`` potential swaps.

            If a fixed small integer (e.g. ``max_neighbors=40``) is set for both
            algorithms, CLARANS tests only 40 single edges and is prone to
            premature stopping (failing 40 consecutive single-edge tests quickly
            after very few swaps, yielding deceptively low runtime but poor inertia),
            while FastCLARANS evaluates ``40 * k`` swap combinations, finding
            many more improving swaps and continuing to optimize deeper.

            For fair comparison and optimal performance, keep the default
            ``'auto'``, which provides equivalent search budgets and ensures
            FastCLARANS is both significantly faster and achieves lower inertia.


    init : {'k-medoids++', 'random', 'heuristic', 'build', array-like}, default='k-medoids++'
        Method for initialization (adapted from scikit-learn-extra). If an
        array-like is provided it should be of shape (n_clusters, n_features)
        and will be snapped to the nearest points in X.

    metric : str or callable, default='euclidean'
        The distance metric to use. Supports all metrics from
        ``sklearn.metrics.pairwise_distances``, ``scipy.spatial.distance.cdist``,
        and ``sklearn.metrics.DistanceMetric`` (e.g., 'euclidean',
        'manhattan', 'cosine', 'chebyshev', 'precomputed') or a callable.

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
    * Schubert, E., & Rousseeuw, P. J. (2021). Fast and eager k-medoids
      clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS
      algorithms. Information Systems, 101, 101804.
    * scikit-learn-extra contributors. KMedoids clustering implementation
      and initialization strategies ('k-medoids++', 'heuristic', 'build').
      https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    """

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        num_local: int = 2,
        max_neighbors: int | str = "auto",
        init: str | ArrayLike = "k-medoids++",
        metric: str | Any = "euclidean",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            num_local=num_local,
            max_neighbors=max_neighbors,
            init=init,
            metric=metric,
            random_state=random_state,
            cost_evaluation="delta",
        )

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
        _warn_cython_unavailable()
        X, random_state, n_samples, n_features = self._validate_input_and_params(X)

        if self.max_neighbors == "auto":
            # FastCLARANS samples 2.5% of non-medoid points per local search
            # (Schubert & Rousseeuw, 2021) instead of 1.25% * k * (n-k) edges.
            # A proportional floor of max(1, 250 // k) guarantees at least 250 edge
            # evaluations (matching Ng & Han 2002) without candidate blowup.
            self.max_neighbors_ = max(
                1,
                int(250 / self.n_clusters),
                int(0.025 * (n_samples - self.n_clusters)),
            )
        else:
            self.max_neighbors_ = int(self.max_neighbors)

        best_cost = np.inf
        best_medoids: np.ndarray | None = None
        best_n_iter = 0
        best_n_swaps = 0

        deterministic_medoids = self._prepare_initial_medoids(X, random_state)
        self._setup_distance_engine(X)

        buf_dtype = (
            np.float32
            if (self.metric == "precomputed" and getattr(X, "dtype", None) == np.float32)
            else np.float64
        )
        d_xc_buf = np.empty(n_samples, dtype=buf_dtype)
        delta_arr_buf = np.zeros(self.n_clusters, dtype=buf_dtype)

        for loc_idx in range(self.num_local):
            current_cost, current_medoids_indices, eval_count, swap_count = (
                self._single_local_search(
                    X, random_state, deterministic_medoids, d_xc_buf, delta_arr_buf
                )
            )

            tol = -max(1e-16, 1e-12 * abs(current_cost))
            if loc_idx == 0 or current_cost < best_cost + tol:
                best_cost = current_cost
                best_medoids = current_medoids_indices.copy()
                best_n_iter = eval_count
                best_n_swaps = swap_count

        if best_medoids is None or not np.isfinite(best_cost):
            raise ValueError(
                f"Clustering failed: all local searches resulted in non-finite cost ({best_cost}). "
                "Check your data for NaNs, infinities, zero vectors with cosine distance, or excessive outliers."
            )

        self.n_iter_ = best_n_iter
        self.n_swaps_ = best_n_swaps

        return self._finalize_fit(X, best_cost, best_medoids)

    def _single_local_search(
        self,
        X: np.ndarray | "spmatrix",
        random_state: np.random.RandomState,
        deterministic_medoids: np.ndarray | None,
        d_xc_buf: np.ndarray | None = None,
        delta_arr_buf: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, int, int]:
        """Perform a single local search from initial medoids to a local optimum."""
        n_samples = X.shape[0]
        if deterministic_medoids is not None:
            current_medoids_indices = deterministic_medoids.copy()
        else:
            current_medoids_indices = self._initialize_medoids(X, random_state)
        current_medoids_indices.sort()

        medoids_dist = self._compute_medoids_distances(X, current_medoids_indices)
        if not medoids_dist.flags.c_contiguous:
            medoids_dist = np.ascontiguousarray(medoids_dist)
        near_idx_map, near_dist, second_dist = self._compute_2min(medoids_dist)
        current_cost: float = float(np.sum(near_dist))

        non_medoid_mask = np.ones(n_samples, dtype=bool)
        non_medoid_mask[current_medoids_indices] = False
        available_candidates = np.flatnonzero(non_medoid_mask)

        i = 0
        swap_count = 0
        eval_count = 0

        while i < self.max_neighbors_:
            eval_count += 1
            if available_candidates.size == 0:
                break

            candidate_idx = int(
                available_candidates[
                    random_state.randint(0, len(available_candidates))
                ]
            )

            cand_row = (
                None
                if self.metric == "precomputed"
                else X[candidate_idx : candidate_idx + 1]
            )
            d_xc = self._compute_1_vs_n(
                cand_row, X, out=d_xc_buf, candidate_idx=candidate_idx
            )

            if self.n_clusters == 1:
                candidate_cost = float(np.sum(d_xc))
                min_delta = candidate_cost - current_cost
                min_delta_idx = 0
            elif (
                _core is not None
                and isinstance(d_xc, np.ndarray)
                and d_xc.flags.c_contiguous
                and d_xc.dtype in (np.float64, np.float32)
                and isinstance(near_dist, np.ndarray)
                and near_dist.flags.c_contiguous
                and isinstance(second_dist, np.ndarray)
                and second_dist.flags.c_contiguous
                and near_dist.dtype == d_xc.dtype
                and isinstance(near_idx_map, np.ndarray)
                and near_idx_map.flags.c_contiguous
            ):
                delta_buf_arg = (
                    delta_arr_buf
                    if (
                        delta_arr_buf is not None
                        and delta_arr_buf.dtype == near_dist.dtype
                    )
                    else None
                )
                best_m, min_delta_val, _ = _core.fastpam1_delta(
                    near_idx_map,
                    near_dist,
                    second_dist,
                    d_xc,
                    n_samples,
                    self.n_clusters,
                    delta_buf_arg,
                )
                min_delta_idx = int(best_m)
                min_delta = float(min_delta_val)
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

            delta_tol = -max(1e-16, 1e-12 * abs(current_cost))
            if min_delta < delta_tol:
                old_medoid = current_medoids_indices[min_delta_idx]
                current_medoids_indices[min_delta_idx] = candidate_idx

                # Incremental distance matrix update in O(1) distance calls
                medoids_dist[:, min_delta_idx] = d_xc
                near_idx_map, near_dist, second_dist = self._compute_2min(medoids_dist)
                current_cost = float(np.sum(near_dist))

                # Update persistent mask on accepted swap
                non_medoid_mask[old_medoid] = True
                non_medoid_mask[candidate_idx] = False
                available_candidates = np.flatnonzero(non_medoid_mask)

                i = 0
                swap_count += 1
            else:
                i += 1

        return current_cost, current_medoids_indices, eval_count, swap_count
