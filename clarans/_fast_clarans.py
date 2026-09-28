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

from ._clarans import CLARANS
from .utils import _core

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


def _fallback_fastpam1_delta(
    near_idx_map: np.ndarray,
    near_dist: np.ndarray,
    second_dist: np.ndarray,
    d_xc: np.ndarray,
    n_clusters: int,
) -> tuple[int, float]:
    """Pure-Python fallback for FastPAM1 delta calculations when Cython is unavailable."""
    removal_loss = np.zeros(n_clusters)
    diff = second_dist - near_dist
    with np.errstate(invalid="ignore"):
        removal_loss += np.bincount(
            near_idx_map, weights=diff, minlength=n_clusters
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
            idx1, weights=term1, minlength=n_clusters
        )

    mask_case2 = (~mask_better_than_nearest) & mask_better_than_second
    term2 = d_xc[mask_case2] - second_dist[mask_case2]
    idx2 = near_idx_map[mask_case2]
    with np.errstate(invalid="ignore"):
        total_delta += np.bincount(
            idx2, weights=term2, minlength=n_clusters
        )

    min_delta_idx = int(np.argmin(total_delta))
    return min_delta_idx, float(total_delta[min_delta_idx])


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

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    random_state : int, RandomState instance or None, default=None
        Controls random number generation for reproducibility.

    verbose : int, default=0
        Verbosity mode. Controls the level of progress messages printed during
        fitting:

        - ``0``: Silent (default).
        - ``1``: Prints progress for each local search iteration (start, completion,
          cost, elapsed time, swaps performed, and candidates evaluated).
        - ``>=2``: Additionally prints details on each successful medoid swap.

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
        sampled per local search before concluding convergence.

    total_neighbors_ : int
        Total number of non-medoid candidate points, equal to
        ``n_samples - n_clusters``.

    n_iter_ : int
        Number of candidate neighbors evaluated during the best local search.

    n_swaps_ : int
        Number of successful medoid swaps performed during the best local
        search.

    total_n_iter_ : int
        Total number of candidate neighbors evaluated across all ``num_local``
        searches.

    total_n_swaps_ : int
        Total number of successful medoid swaps performed across all
        ``num_local`` searches.

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
        metric_params: dict[str, Any] | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            num_local=num_local,
            max_neighbors=max_neighbors,
            init=init,
            metric=metric,
            metric_params=metric_params,
            random_state=random_state,
            cost_evaluation="delta",
            verbose=verbose,
        )

    def _init_search_budget(self, n_samples: int) -> None:
        """Initialize max_neighbors_ and total_neighbors_ search budget for FastCLARANS."""
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

        self.total_neighbors_ = n_samples - self.n_clusters

    def _allocate_search_buffers(
        self, X: Any, n_samples: int
    ) -> dict[str, np.ndarray]:
        """Pre-allocate reusable scratch buffers for neighbor evaluations."""
        bufs = super()._allocate_search_buffers(X, n_samples)
        buf_dtype = bufs["d_xc_buf"].dtype
        bufs["delta_arr_buf"] = np.zeros(self.n_clusters, dtype=buf_dtype)
        return bufs

    def _call_single_local_search(
        self,
        X: np.ndarray | "spmatrix",
        random_state: np.random.RandomState,
        deterministic_medoids: np.ndarray | None,
        buffers: dict[str, np.ndarray],
        loc_idx: int = 1,
    ) -> tuple[float, np.ndarray, int, int]:
        """Dispatch to _single_local_search with FastPAM1 buffers."""
        try:
            return self._single_local_search(
                X,
                random_state,
                deterministic_medoids,
                buffers["d_xc_buf"],
                buffers["delta_arr_buf"],
                loc_idx=loc_idx,
            )
        except TypeError:
            return self._single_local_search(
                X,
                random_state,
                deterministic_medoids,
                buffers["d_xc_buf"],
                buffers["delta_arr_buf"],
            )

    def _single_local_search(
        self,
        X: np.ndarray | "spmatrix",
        random_state: np.random.RandomState,
        deterministic_medoids: np.ndarray | None,
        d_xc_buf: np.ndarray | None = None,
        delta_arr_buf: np.ndarray | None = None,
        loc_idx: int = 1,
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

        if self.verbose >= 2:
            r_idx = loc_idx
            print(
                f"  Restart {r_idx}/{self.num_local} (init cost: {current_cost:.5f}):"
            )

        non_medoid_mask = np.ones(n_samples, dtype=bool)
        non_medoid_mask[current_medoids_indices] = False
        available_candidates = np.flatnonzero(non_medoid_mask)

        # Pre-evaluate Cython kernel availability and pre-configure delta buffer
        can_use_cython = self._can_use_cython(
            d_xc_buf, near_idx_map, near_dist, second_dist
        )
        delta_buf_arg = (
            delta_arr_buf
            if (
                can_use_cython
                and delta_arr_buf is not None
                and delta_arr_buf.dtype == near_dist.dtype
            )
            else None
        )

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
            elif can_use_cython:
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
                min_delta_idx, min_delta = _fallback_fastpam1_delta(
                    near_idx_map, near_dist, second_dist, d_xc, self.n_clusters
                )

            delta_tol = self._delta_tolerance(current_cost)
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
                if self.verbose >= 2:
                    print(
                        f"      swap {swap_count:3d} | eval {eval_count:5d} | "
                        f"cost {current_cost:14.5f} | diff {min_delta:12.5f}"
                    )
            else:
                i += 1

        return current_cost, current_medoids_indices, eval_count, swap_count
