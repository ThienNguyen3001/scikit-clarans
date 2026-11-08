"""FastCLARANS implementation.

Provides a faster variant of CLARANS by using FastPAM1 delta-cost updates.
This implementation computes distances on-the-fly as recommended in the
original paper, making it memory-efficient for large datasets while still
benefiting from the O(k) speedup per swap evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._clarans import CLARANS
from .utils import _core


def _fallback_fastpam1_delta(
    near_idx_map: np.ndarray,
    near_dist: np.ndarray,
    second_dist: np.ndarray,
    d_xc: np.ndarray,
    n_clusters: int,
) -> tuple[int, float]:
    """Pure-Python fallback for FastPAM1 delta calculations when Cython is unavailable."""
    mask_closer = d_xc < near_dist
    delta_td = float(np.sum(d_xc[mask_closer] - near_dist[mask_closer]))

    weights = np.where(
        mask_closer,
        0.0,
        np.where(d_xc < second_dist, d_xc - near_dist, second_dist - near_dist),
    )
    delta_arr = np.bincount(near_idx_map, weights=weights, minlength=n_clusters) + delta_td
    best_m = int(np.argmin(delta_arr))
    return best_m, float(delta_arr[best_m])


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

    max_neighbors : int, str or None, default=None
        The maximum number of non-medoid candidates to sample per local
        search. If ``None`` (default), defaults to ``max(1, int(250 / k), int(0.025 * (n - k)))``
        combining the 2.5% non-medoid sample recommended by Schubert & Rousseeuw
        (2021) with a proportional floor equivalent to 250 edge evaluations
        (Ng & Han, 2002). This adaptive default automatically scales with dataset
        size without requiring manual tuning.
        The option ``'auto'`` is deprecated in version 0.4.1 and will be removed
        in version 0.5.0; use ``None`` instead.


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
            ``None``, which provides equivalent search budgets and ensures
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

    n_jobs : int, default=None
        The number of parallel jobs to run across ``num_local`` local search
        restarts during ``fit``. ``None`` means 1 unless in a
        ``joblib.parallel_backend`` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    cluster_centers_ : {ndarray, sparse matrix} of shape (n_clusters, n_features)
        Coordinates of cluster centers (medoids). Defined only when
        ``metric != 'precomputed'`` (precomputed distance matrices do not carry
        feature coordinates; use ``medoid_indices_`` instead).

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

    neighborhood_size_ : int
        Total number of non-medoid candidate points, equal to
        ``n_samples - n_clusters``.

    n_iter_ : int
        Number of candidate neighbors evaluated during the best local search
        restart.

    n_swaps_ : int
        Number of successful medoid swaps performed during the best local search
        restart.

    n_features_in_ : int
        Number of features seen during :term:`fit`. For ``metric='precomputed'``,
        this equals the number of training samples (the column dimension of the
        distance matrix).

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
      algorithms. Information Systems, 101, 101804. The original reference
      implementation in Java is part of the ELKI data mining framework.
    * scikit-learn-extra contributors. KMedoids clustering implementation
      and initialization strategies ('k-medoids++', 'heuristic', 'build').
      https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    """

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        num_local: int = 2,
        max_neighbors: int | str | None = None,
        init: str | ArrayLike = "k-medoids++",
        metric: str | Any = "euclidean",
        metric_params: dict[str, Any] | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
        n_jobs: int | None = None,
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
            n_jobs=n_jobs,
        )

    def _init_search_budget(self, n_samples: int) -> None:
        """Initialize max_neighbors_ and neighborhood_size_ search budget for FastCLARANS."""
        if self.max_neighbors is None or self.max_neighbors == "auto":
            # Sample 2.5% of non-medoid points (Schubert & Rousseeuw, 2021)
            # with a floor of 250 // k to evaluate at least 250 edges per restart.
            self.max_neighbors_ = max(
                1,
                int(250 / self.n_clusters),
                int(0.025 * (n_samples - self.n_clusters)),
            )
        else:
            self.max_neighbors_ = int(self.max_neighbors)

        self.neighborhood_size_ = n_samples - self.n_clusters

    def _sample_candidate(
        self,
        available_candidates: np.ndarray,
        random_state: np.random.RandomState,
    ) -> tuple[int, int | None]:
        """Sample a non-medoid candidate at random for FastCLARANS."""
        candidate_idx = int(
            available_candidates[random_state.randint(0, len(available_candidates))]
        )
        return candidate_idx, None

    def _evaluate_candidate_swap(
        self,
        d_xc: np.ndarray,
        near_idx_map: np.ndarray | None,
        near_dist: np.ndarray | None,
        second_dist: np.ndarray | None,
        current_cost: float,
        can_use_cython: bool,
        n_samples: int,
        medoid_pos: int | None = None,
        delta_arr_buf: np.ndarray | None = None,
    ) -> tuple[int, float]:
        """Evaluate candidate swap against all k medoids simultaneously via FastPAM1."""
        if self.n_clusters == 1:
            candidate_cost = float(np.sum(d_xc))
            return 0, candidate_cost - current_cost
        elif can_use_cython:
            assert near_idx_map is not None
            assert near_dist is not None
            assert second_dist is not None
            best_m, min_delta_val, _ = _core.fastpam1_delta(
                near_idx_map,
                near_dist,
                second_dist,
                d_xc,
                n_samples,
                self.n_clusters,
                delta_arr_buf,
            )
            return int(best_m), float(min_delta_val)
        else:
            assert near_idx_map is not None
            assert near_dist is not None
            assert second_dist is not None
            return _fallback_fastpam1_delta(
                near_idx_map, near_dist, second_dist, d_xc, self.n_clusters
            )
