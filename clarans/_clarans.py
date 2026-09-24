from __future__ import annotations

import inspect
import time
import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import DistanceMetric, pairwise_distances_argmin_min, pairwise_distances
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from ._initialization import (
    initialize_build,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)
from .utils import (
    _SCIPY_METRIC_MAP,
    _warn_cython_unavailable,
    calculate_cost,
    check_medoids,
)

# Detect whether check_array expects 'ensure_all_finite' (scikit-learn >= 1.6)
# or 'force_all_finite' (< 1.6)
_FINITE_PARAM = (
    "ensure_all_finite"
    if "ensure_all_finite" in inspect.signature(check_array).parameters
    else "force_all_finite"
)


def _finite_kwarg(allow_nan: bool) -> dict[str, Any]:
    return {_FINITE_PARAM: "allow-nan" if allow_nan else True}


try:
    _DM_METRICS = {m for m in DistanceMetric.get_valid_metric_ids() if m != "pyfunc"}
except (AttributeError, TypeError):
    _DM_METRICS = {
        "braycurtis", "canberra", "chebyshev", "cityblock", "correlation",
        "cosine", "dice", "euclidean", "hamming", "haversine", "infinity",
        "jaccard", "kulsinski", "l1", "l2", "mahalanobis", "manhattan",
        "matching", "minkowski", "p", "rogerstanimoto", "russellrao",
        "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule",
    }

_CDIST_EXTRA_METRICS = {"jensenshannon"}
_EXCLUDED_METRICS = {"kulczynski1", "wminkowski"}

_ALL_VALID_METRICS = frozenset(
    (
        set(PAIRWISE_DISTANCE_FUNCTIONS.keys())
        | _DM_METRICS
        | _CDIST_EXTRA_METRICS
        | {"precomputed"}
        | set(_SCIPY_METRIC_MAP.keys())
    )
    - _EXCLUDED_METRICS
)

try:
    from . import _core
except ImportError:
    _core = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


class CLARANS(ClusterMixin, TransformerMixin, BaseEstimator):
    """Clustering Large Applications based on RANdomized Search (CLARANS).

    CLARANS is an efficient k-medoids clustering algorithm that searches for
    optimal medoids by examining a randomized subset of neighboring medoid
    configurations on the graph G_{n, k}.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form (also the number of medoids to
        generate).

    num_local : int, default=2
        The number of local searches to perform.
        CLARANS runs the search process ``num_local`` times starting from
        different random nodes to reduce the chance of getting stuck in
        poor local minima. Increasing this improves solution quality but
        increases runtime.

    max_neighbors : int or 'auto', default='auto'
        The maximum number of neighbors (random swaps) to examine during
        each step. If ``'auto'``, it defaults to ``max(250, 1.25% of k*(n-k))``
        as recommended in the original paper (Ng & Han, 2002). This adaptive
        default balances runtime and solution quality automatically without
        requiring manual tuning. Higher values make the algorithm behave more
        like PAM (checking more neighbors); lower values make it faster.

        .. note::
            In CLARANS, each neighbor evaluation tests a single randomly chosen
            pair ``(medoid, candidate)``. Setting a very small integer (e.g.
            ``max_neighbors=40``) checks only a tiny fraction of the
            ``k * (n - k)`` search space, which often causes CLARANS to fail
            consecutive tests early and stop prematurely after only a few swaps.
            Contrast this with ``FastCLARANS``, where each candidate step checks
            all ``k`` medoids simultaneously via FastPAM1 delta caching.


    init : {'k-medoids++', 'random', 'heuristic', 'build', array-like}, default='k-medoids++'
        Strategy for selecting initial medoids (adapted from scikit-learn-extra):

        - ``'k-medoids++'``: Optimized probabilistic initialization (similar
          to k-means++) for faster convergence.
        - ``'random'``: Selects ``n_clusters`` random points. Fast but can
          result in poor starting points.
        - ``'heuristic'``: Selects points that are "central" to the data
          (minimizing distance to all others).
        - ``'build'``: The greedy initialization from the original PAM
          algorithm. High quality but slow (O(N^2)).
        - array-like: If an array is passed, it should be of shape
          ``(n_clusters, n_features)`` giving the initial centers, or
          ``(n_clusters,)`` of integer indices if ``metric='precomputed'``.

    metric : str or callable, default='euclidean'
        The distance metric to use. Supports all metrics from
        ``sklearn.metrics.pairwise_distances``, ``scipy.spatial.distance.cdist``,
        and ``sklearn.metrics.DistanceMetric`` (e.g., 'euclidean',
        'manhattan', 'cosine', 'chebyshev', 'precomputed') or a callable.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for medoid swaps and random
        initialization. Pass an int for reproducible output across multiple
        function calls.

    cost_evaluation : {'delta', 'brute_force'}, default='delta'
        Strategy for evaluating candidate neighbor medoid swaps:

        - ``'delta'``: Uses FastPAM1-style nearest and second-nearest distance
          tracking (d1, d2) to evaluate swaps in O(n*d) without recalculating
          the full clustering cost. Recommended for optimal performance.
        - ``'brute_force'``: Recalculates the total clustering cost from scratch
          in O(n*k*d) at each candidate swap, following the original classic
          CLARANS algorithm (Ng & Han, 2002).

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
        Labels of each point.

    medoid_indices_ : ndarray of shape (n_clusters,)
        Indices of the medoids in the training set X.

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    max_neighbors_ : int
        Effective maximum number of non-improving neighbors examined per
        local search.

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
    - Time complexity: In each local search, random candidate neighbor swaps
      are evaluated on the search graph G_{n,k}. The search terminates when
      ``max_neighbors`` consecutive non-improving swaps are tested. With S
      successful swaps and distance evaluation cost O(n * k * d), runtime per
      local search is bounded by O((S + max_neighbors) * n * k * d), repeated
      ``num_local`` times.
    - Memory complexity: Distances are computed on-the-fly, keeping memory
      usage at O(n) instead of O(n^2). Note that initialization methods
      such as ``'build'`` and ``'heuristic'`` compute full pairwise distance
      matrices and therefore have O(n^2) time and memory costs.
    - Compared with ``FastCLARANS``: This class implements the classic
      randomized search from Ng & Han (2002). For faster execution on larger
      datasets, consider ``FastCLARANS``, which tests swaps with all k medoids
      simultaneously using FastPAM1 delta calculations.
    
    References
    ----------
    * Ng, R. T., & Han, J. (2002). CLARANS: A method for clustering objects for
      spatial data mining. IEEE Transactions on Knowledge and Data Engineering,
      14(5), 1003-1016.
    * scikit-learn-extra contributors. KMedoids clustering implementation
      and initialization strategies ('k-medoids++', 'heuristic', 'build').
      https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html

    Examples
    --------
    >>> from clarans import CLARANS
    >>> model = CLARANS(n_clusters=3, random_state=0)
    >>> model.fit(X)
    """

    def __init__(
        self,
        *,
        n_clusters=8,
        num_local=2,
        max_neighbors="auto",
        init="k-medoids++",
        metric="euclidean",
        metric_params=None,
        random_state=None,
        cost_evaluation="delta",
        verbose=0,
    ):
        self.n_clusters = n_clusters
        self.num_local = num_local
        self.max_neighbors = max_neighbors
        self.init = init
        self.metric = metric
        self.metric_params = metric_params
        self.random_state = random_state
        self.cost_evaluation = cost_evaluation
        self.verbose = verbose

    def _prepare_initial_medoids(self, X, random_state):
        """Pre-compute initial medoids if the initialization strategy is deterministic.

        For deterministic strategies ('build', 'heuristic', or explicit array),
        computing initial medoids once before the ``num_local`` loop avoids costly
        O(N^2) recalculations across local search restarts.
        """
        is_deterministic = (
            (isinstance(self.init, str) and self.init in ("build", "heuristic"))
            or not isinstance(self.init, str)
        )
        if is_deterministic:
            if self.num_local > 1:
                if isinstance(self.init, str):
                    warnings.warn(
                        f"The '{self.init}' initialization is deterministic, so all "
                        f"{self.num_local} local searches start from the exact same initial "
                        f"medoids. While randomized neighbor sampling still explores "
                        f"different search paths, consider 'k-medoids++' for diverse "
                        f"starting points or num_local=1 to save computation.",
                        UserWarning,
                    )
                else:
                    warnings.warn(
                        f"An explicit init array was provided, so all {self.num_local} "
                        f"local searches start from the exact same initial medoids. "
                        f"While randomized neighbor sampling still explores different "
                        f"search paths, consider 'k-medoids++' for diverse starting points "
                        f"or num_local=1 to save computation.",
                        UserWarning,
                    )
            return self._initialize_medoids(X, random_state)
        return None

    def _initialize_medoids(self, X, random_state):
        """Select initial medoid indices according to ``self.init``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The validated data matrix.

        random_state : RandomState
            The random state instance to use.

        Returns
        -------
        current_medoids_indices : ndarray of shape (n_clusters,)
            Indices of the initial medoids.
        """
        n_samples, n_features = X.shape
        all_indices = np.arange(n_samples)

        if isinstance(self.init, str):
            if self.init == "random":
                current_medoids_indices = random_state.choice(
                    n_samples, self.n_clusters, replace=False
                )
            elif self.init == "k-medoids++":
                current_medoids_indices = initialize_k_medoids_plus_plus(
                    X, self.n_clusters, random_state, self.metric, metric_params=self.metric_params
                )
            elif self.init == "heuristic":
                current_medoids_indices = initialize_heuristic(
                    X, self.n_clusters, self.metric, metric_params=self.metric_params
                )
            elif self.init == "build":
                current_medoids_indices = initialize_build(
                    X, self.n_clusters, self.metric, metric_params=self.metric_params
                )
            else:
                raise ValueError(f"Unknown init method: {self.init!r}")
        else:
            try:
                init_arr = np.asarray(self.init)
            except Exception as err:
                raise ValueError(f"Could not convert init to array: {err}") from err

            if self.metric == "precomputed":
                if (
                    init_arr.ndim == 1
                    and len(init_arr) == self.n_clusters
                    and np.issubdtype(init_arr.dtype, np.integer)
                ):
                    current_medoids_indices = check_medoids(init_arr, n_samples=n_samples)
                    return np.array(current_medoids_indices, dtype=int)

                init_centers = check_array(self.init)
                if init_centers.shape != (self.n_clusters, n_features):
                    raise ValueError(
                        f"init array must be of shape ({self.n_clusters}, {n_features})"
                    )
                current_medoids_indices = np.argmin(init_centers, axis=1)
            else:
                try:
                    init_centers = check_array(self.init)
                except ValueError as err:
                    if "Expected 2D array, got 1D array instead" in str(err):
                        raise ValueError(
                            f"init array must be 2D of shape ({self.n_clusters}, {n_features}). "
                            "If you want to specify medoid indices, use metric='precomputed' "
                            "or pass X[indices] as initial centers."
                        ) from err
                    raise
                if init_centers.shape != (self.n_clusters, n_features):
                    raise ValueError(
                        f"init array must be of shape ({self.n_clusters}, {n_features})"
                    )

                params = self.metric_params if self.metric_params is not None else {}
                if not issparse(X) and not issparse(init_centers):
                    scipy_metric = _SCIPY_METRIC_MAP.get(self.metric, self.metric)
                    try:
                        D = cdist(init_centers, X, metric=scipy_metric, **params)
                        current_medoids_indices = np.argmin(D, axis=1)
                    except Exception:
                        current_medoids_indices, _ = pairwise_distances_argmin_min(
                            init_centers, X, metric=self.metric, metric_kwargs=params
                        )
                else:
                    current_medoids_indices, _ = pairwise_distances_argmin_min(
                        init_centers, X, metric=self.metric, metric_kwargs=params
                    )

            current_medoids_indices = np.array(current_medoids_indices, dtype=int)

            current_medoids_indices = np.unique(current_medoids_indices)

            if len(current_medoids_indices) < self.n_clusters:
                warnings.warn(
                    "Provided init centers map to duplicate points in X. "
                    "Filling duplicates with random points."
                )
                remaining = self.n_clusters - len(current_medoids_indices)
                available = np.setdiff1d(
                    all_indices, current_medoids_indices, assume_unique=True
                )

                if len(available) < remaining:
                    raise ValueError(
                        "Not enough unique points to fill up to n_clusters."
                    )

                fillers = random_state.choice(available, remaining, replace=False)
                current_medoids_indices = np.concatenate(
                    [current_medoids_indices, fillers]
                )

        return np.array(current_medoids_indices, dtype=int)

    def __sklearn_tags__(self):
        """Declare estimator capabilities for scikit-learn's check_estimator.

        CLARANS can accept CSR/CSC sparse matrices as input.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = (self.metric == "nan_euclidean")
        if self.metric == "precomputed":
            tags.input_tags.pairwise = True
        return tags

    def _more_tags(self):
        return {
            "pairwise": self.metric == "precomputed",
            "allow_nan": self.metric == "nan_euclidean",
        }

    def fit(self, X: ArrayLike | "spmatrix", y: Any = None) -> "CLARANS":
        """
        Compute CLARANS clustering.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training instances to cluster. Accepts CSR/CSC sparse matrices.

        y : Ignored, default=None
            Not used, present here for API consistency with scikit-learn
            pipelines and ClusterMixin.

        Returns
        -------
        self : CLARANS
            Fitted estimator. Attributes set on the estimator include
            ``medoid_indices_``, ``cluster_centers_``, and ``labels_``.

        Raises
        ------
        ValueError
            If ``n_clusters >= n_samples``, or if an explicit ``init`` array
            has an incompatible shape, or if not enough unique points exist
            to initialize the requested number of clusters.

        Notes
        -----
        - Time complexity: each local search evaluates up to ``max_neighbors``
          candidate swaps, and each cost evaluation is O(n * k) (distance
          to medoids), so the worst-case runtime is roughly
          O(num_local * max_neighbors * n * k).
        - Initialization methods such as ``'heuristic'`` and ``'build'``
          may compute the full pairwise distance matrix and therefore have
          O(n^2) time and memory costs.
        - Compared with ``FastCLARANS``, this implementation avoids
          caching the full distance matrix and is more memory-friendly for
          very large datasets at the cost of repeated distance computations.

        Examples
        --------
        >>> from clarans import CLARANS
        >>> model = CLARANS(n_clusters=3, random_state=0)
        >>> model.fit(X)
        """
        _warn_cython_unavailable()
        X, random_state, n_samples, n_features = self._validate_input_and_params(X)

        if self.max_neighbors == "auto":
            self.max_neighbors_ = max(
                250, int(0.0125 * self.n_clusters * (n_samples - self.n_clusters))
            )
        else:
            self.max_neighbors_ = int(self.max_neighbors)

        best_cost = np.inf
        best_medoids = None
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

        if self.verbose:
            print(
                f"[CLARANS] Fitting with n_clusters={self.n_clusters}, "
                f"num_local={self.num_local}, max_neighbors={self.max_neighbors_}"
            )

        start_fit_time = time.perf_counter()

        try:
            for loc_idx in range(self.num_local):
                if self.verbose:
                    print(f"[CLARANS] Local search {loc_idx + 1}/{self.num_local}:")
                loc_start_time = time.perf_counter()

                current_cost, current_medoids_indices, eval_count, swap_count = (
                    self._single_local_search(
                        X, random_state, deterministic_medoids, d_xc_buf
                    )
                )
                loc_elapsed = time.perf_counter() - loc_start_time

                if self.verbose:
                    print(
                        f"[CLARANS] Local search {loc_idx + 1}/{self.num_local} done "
                        f"in {loc_elapsed:.3f}s (Cost: {current_cost:.5f}, "
                        f"Swaps: {swap_count}, Evaluated: {eval_count})"
                    )

                tol = -max(1e-16, 1e-12 * abs(current_cost))
                if (
                    loc_idx == 0
                    or not np.isfinite(best_cost)
                    or (np.isfinite(current_cost) and current_cost < best_cost + tol)
                ):
                    best_cost = current_cost
                    best_medoids = current_medoids_indices.copy()
                    best_n_iter = eval_count
                    best_n_swaps = swap_count

            if best_medoids is None or not np.isfinite(best_cost):
                raise ValueError(
                    "Clustering failed: all local search iterations produced non-finite "
                    "costs (inf or NaN). Check input data or distance metric."
                )

            self.n_iter_ = best_n_iter
            self.n_swaps_ = best_n_swaps

            if self.verbose:
                total_elapsed = time.perf_counter() - start_fit_time
                print(
                    f"[CLARANS] Best cost: {best_cost:.5f} achieved in {total_elapsed:.3f}s"
                )

            return self._finalize_fit(X, best_cost, best_medoids)
        finally:
            if hasattr(self, "_precomputed_source"):
                del self._precomputed_source

    def _single_local_search(
        self,
        X: np.ndarray | "spmatrix",
        random_state: np.random.RandomState,
        deterministic_medoids: np.ndarray | None,
        d_xc_buf: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, int, int]:
        """Perform a single local search from initial medoids to a local optimum."""
        n_samples = X.shape[0]
        if deterministic_medoids is not None:
            current_medoids_indices = deterministic_medoids.copy()
        else:
            current_medoids_indices = self._initialize_medoids(X, random_state)

        if self.cost_evaluation == "delta":
            medoids_dist = self._compute_medoids_distances(X, current_medoids_indices)
            if not medoids_dist.flags.c_contiguous:
                medoids_dist = np.ascontiguousarray(medoids_dist)
            near_idx_map, near_dist, second_dist = self._compute_2min(medoids_dist)
            current_cost = float(np.sum(near_dist))
        else:
            medoids_dist = None
            current_cost = calculate_cost(
                X, current_medoids_indices, self.metric, metric_params=self.metric_params
            )
            near_idx_map = None
            near_dist = None
            second_dist = None

        # Maintain persistent non-medoid mask across iterations
        non_medoid_mask = np.ones(n_samples, dtype=bool)
        non_medoid_mask[current_medoids_indices] = False
        available_candidates = np.flatnonzero(non_medoid_mask)

        i = 0
        swap_count = 0
        eval_count = 0

        while i < self.max_neighbors_:
            eval_count += 1
            random_medoid_pos = int(random_state.randint(0, self.n_clusters))

            if available_candidates.size == 0:
                break

            # Fast direct index draw matching random_state.choice 100% bit-exact
            random_non_medoid_candidate = int(
                available_candidates[
                    random_state.randint(0, len(available_candidates))
                ]
            )

            if self.cost_evaluation == "delta":
                cand_row = (
                    None
                    if self.metric == "precomputed"
                    else X[
                        random_non_medoid_candidate : random_non_medoid_candidate + 1
                    ]
                )
                d_xc = self._compute_1_vs_n(
                    cand_row, X, out=d_xc_buf, candidate_idx=random_non_medoid_candidate
                )

                if self.n_clusters == 1:
                    candidate_cost = float(np.sum(d_xc))
                    total_delta = candidate_cost - current_cost
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
                    total_delta = float(
                        _core.clarans_delta(
                            near_idx_map,
                            near_dist,
                            second_dist,
                            d_xc,
                            random_medoid_pos,
                            n_samples,
                        )
                    )
                else:
                    assert near_idx_map is not None
                    assert near_dist is not None
                    assert second_dist is not None
                    is_assigned_to_m = near_idx_map == random_medoid_pos
                    delta_assigned = (
                        np.minimum(
                            second_dist[is_assigned_to_m], d_xc[is_assigned_to_m]
                        )
                        - near_dist[is_assigned_to_m]
                    )
                    delta_others = np.minimum(
                        0.0,
                        d_xc[~is_assigned_to_m] - near_dist[~is_assigned_to_m],
                    )
                    total_delta = float(
                        np.sum(delta_assigned) + np.sum(delta_others)
                    )

                delta_tol = -max(1e-16, 1e-12 * abs(current_cost))
                if total_delta < delta_tol:
                    old_medoid = current_medoids_indices[random_medoid_pos]
                    current_medoids_indices[random_medoid_pos] = (
                        random_non_medoid_candidate
                    )

                    # Incremental update: update only the swapped column in O(1) distance calls
                    assert medoids_dist is not None
                    medoids_dist[:, random_medoid_pos] = d_xc
                    near_idx_map, near_dist, second_dist = self._compute_2min(medoids_dist)
                    current_cost = float(np.sum(near_dist))

                    # Update persistent mask on accepted swap
                    non_medoid_mask[old_medoid] = True
                    non_medoid_mask[random_non_medoid_candidate] = False
                    available_candidates = np.flatnonzero(non_medoid_mask)

                    i = 0
                    swap_count += 1
                    if self.verbose >= 2:
                        print(
                            f"  Swap {swap_count:4d} | Evaluated: {eval_count:6d} | "
                            f"Cost: {current_cost:.5f} | Delta: {total_delta:.5f}"
                        )
                else:
                    i += 1
            else:
                neighbor_medoids_indices = current_medoids_indices.copy()
                neighbor_medoids_indices[random_medoid_pos] = (
                    random_non_medoid_candidate
                )

                neighbor_cost = calculate_cost(
                    X,
                    neighbor_medoids_indices,
                    self.metric,
                    metric_params=self.metric_params,
                )

                delta_tol = -max(1e-16, 1e-12 * abs(current_cost))
                if neighbor_cost < current_cost + delta_tol:
                    old_medoid = current_medoids_indices[random_medoid_pos]
                    current_medoids_indices = neighbor_medoids_indices
                    delta_cost = neighbor_cost - current_cost
                    current_cost = neighbor_cost

                    non_medoid_mask[old_medoid] = True
                    non_medoid_mask[random_non_medoid_candidate] = False
                    available_candidates = np.flatnonzero(non_medoid_mask)

                    i = 0
                    swap_count += 1
                    if self.verbose >= 2:
                        print(
                            f"  Swap {swap_count:4d} | Evaluated: {eval_count:6d} | "
                            f"Cost: {current_cost:.5f} | Delta: {delta_cost:.5f}"
                        )
                else:
                    i += 1

        return current_cost, current_medoids_indices, eval_count, swap_count

    def _setup_distance_engine(self, X: np.ndarray | "spmatrix") -> None:
        """Determine the most efficient distance calculation engine.

        Priority:
        1. 'precomputed': distance matrix is already computed.
        2. 'cdist': SciPy C-kernel for dense arrays (Euclidean, Manhattan, Chebyshev, etc.).
        3. 'distance_metric': Scikit-Learn DistanceMetric for sparse matrices or callables.
        4. 'pairwise': Fallback using pairwise_distances.
        """
        if self.metric == "precomputed":
            self._dist_engine = "precomputed"
            self._scipy_metric = None
            self._dm_instance = None
            if not issparse(X):
                n_s = X.shape[0]
                if (
                    _core is not None
                    and hasattr(_core, "is_matrix_symmetric")
                    and isinstance(X, np.ndarray)
                    and X.flags.c_contiguous
                    and X.dtype in (np.float64, np.float32)
                ):
                    is_sym = _core.is_matrix_symmetric(X, n_s, 1e-5, 1e-8)
                else:
                    is_sym = bool(np.allclose(X, X.T))

                if is_sym:
                    self._precomputed_source = (
                        X if (isinstance(X, np.ndarray) and X.flags.c_contiguous)
                        else np.ascontiguousarray(X)
                    )
                    self._precomputed_is_sym = True
                else:
                    self._precomputed_source = np.ascontiguousarray(X.T)
                    self._precomputed_is_sym = False
            else:
                self._precomputed_source = X
                self._precomputed_is_sym = False
            return

        params = self.metric_params if self.metric_params is not None else {}

        # 1. Try SciPy cdist for dense NumPy array
        if isinstance(X, np.ndarray) and not issparse(X) and isinstance(self.metric, str):
            mapped_metric = _SCIPY_METRIC_MAP.get(self.metric, self.metric)
            try:
                cdist(X[:1], X[:1], metric=mapped_metric, **params)
                self._dist_engine = "cdist"
                self._scipy_metric = mapped_metric
                self._dm_instance = None
                return
            except Exception:
                pass

        # 2. Try Scikit-Learn DistanceMetric (supports CSR sparse matrix & callable functions)
        try:
            self._dm_instance = DistanceMetric.get_metric(self.metric, **params)
            self._dm_instance.pairwise(X[:1], X[:1])
            self._dist_engine = "distance_metric"
            self._scipy_metric = None
            return
        except Exception:
            pass

        # 3. Fallback
        self._dist_engine = "pairwise"
        self._scipy_metric = None
        self._dm_instance = None

    def _compute_1_vs_n(
        self,
        cand_row: Any,
        X: np.ndarray | "spmatrix",
        out: np.ndarray | None = None,
        candidate_idx: int | None = None,
    ) -> np.ndarray:
        """Compute distances from all samples in X to a single candidate sample."""
        engine = getattr(self, "_dist_engine", "pairwise")
        params = self.metric_params if self.metric_params is not None else {}
        if engine == "cdist":
            if out is not None and out.dtype == np.float64:
                cdist(
                    cand_row,
                    X,
                    metric=self._scipy_metric,
                    out=out.reshape(1, -1),
                    **params,
                )
                return out
            return cdist(cand_row, X, metric=self._scipy_metric, **params)[0]
        elif engine == "precomputed":
            if candidate_idx is not None:
                source = getattr(self, "_precomputed_source", X)
                if isinstance(source, np.ndarray):
                    row_data = source[candidate_idx]
                    if (
                        out is not None
                        and isinstance(out, np.ndarray)
                        and out.shape[0] == row_data.shape[0]
                    ):
                        np.copyto(out, row_data)
                        return out
                    return row_data.copy()
                col_data = (
                    source[candidate_idx]
                    if getattr(self, "_precomputed_is_sym", False)
                    else X[:, candidate_idx]
                )
                res = (
                    col_data.toarray().ravel()
                    if hasattr(col_data, "toarray")
                    else np.asarray(col_data).ravel()
                )
                if (
                    out is not None
                    and isinstance(out, np.ndarray)
                    and out.shape[0] == res.shape[0]
                ):
                    np.copyto(out, res)
                    return out
                return res
            col_arr = (
                cand_row.toarray().ravel()
                if hasattr(cand_row, "toarray")
                else np.asarray(cand_row).ravel()
            )
            if (
                out is not None
                and isinstance(out, np.ndarray)
                and out.shape[0] == col_arr.shape[0]
            ):
                np.copyto(out, col_arr)
                return out
            return col_arr
        elif engine == "distance_metric" and self._dm_instance is not None:
            res = self._dm_instance.pairwise(X, cand_row).ravel()
            if out is not None:
                np.copyto(out, res)
                return out
            return res
        else:
            res = pairwise_distances(
                X, cand_row, metric=self.metric, **params
            ).ravel()
            if out is not None:
                np.copyto(out, res)
                return out
            return res

    def _compute_medoids_distances(
        self, X: np.ndarray | "spmatrix", medoids_indices: Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """Compute distances from all samples in X to the given medoids."""
        if self.metric == "precomputed":
            sub_mat = X[:, medoids_indices]
            return (
                sub_mat.toarray()
                if hasattr(sub_mat, "toarray")
                else np.asarray(sub_mat)
            )
        else:
            medoids = X[medoids_indices]
            engine = getattr(self, "_dist_engine", "pairwise")
            params = self.metric_params if self.metric_params is not None else {}
            if engine == "cdist" and isinstance(X, np.ndarray) and not issparse(X):
                return cdist(X, medoids, metric=self._scipy_metric, **params)
            elif engine == "distance_metric" and self._dm_instance is not None:
                return self._dm_instance.pairwise(X, medoids)
            else:
                return pairwise_distances(X, medoids, metric=self.metric, **params)

    def _compute_2min(
        self, subD: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute nearest and second-nearest medoid info from subD distance matrix."""
        n_samples = subD.shape[0]
        if self.n_clusters >= 2:
            if (
                _core is not None
                and isinstance(subD, np.ndarray)
                and subD.flags.c_contiguous
                and subD.dtype in (np.float64, np.float32)
            ):
                return _core.update_cache_2min(subD, n_samples, self.n_clusters)

            # argpartition(., 1) is O(n*k) vs full sort O(n*k*log k).
            # For kth=1: position 0 holds the smallest, position 1 the 2nd smallest.
            part_idx = np.argpartition(subD, 1, axis=1)[:, :2]
            smallest_idx = part_idx[:, 0]
            second_smallest_idx = part_idx[:, 1]

            near_dist = subD[np.arange(n_samples), smallest_idx]
            second_dist = subD[np.arange(n_samples), second_smallest_idx]
            near_idx_map = smallest_idx
        else:
            near_dist = subD[:, 0]
            second_dist = np.full(n_samples, np.inf)
            near_idx_map = np.zeros(n_samples, dtype=int)

        return near_idx_map, near_dist, second_dist

    def _update_cache(
        self, X: np.ndarray | "spmatrix", medoids_indices: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute nearest and second-nearest medoid information on-the-fly."""
        subD = self._compute_medoids_distances(X, medoids_indices)
        if not subD.flags.c_contiguous:
            subD = np.ascontiguousarray(subD)
        return self._compute_2min(subD)

    def _validate_input_and_params(self, X):
        """Validate estimator parameters and input data array."""
        if (
            not isinstance(self.n_clusters, (int, np.integer))
            or isinstance(self.n_clusters, (bool, np.bool_))
            or self.n_clusters < 1
        ):
            raise ValueError(f"n_clusters must be >= 1; got {self.n_clusters}")
        if (
            not isinstance(self.num_local, (int, np.integer))
            or isinstance(self.num_local, (bool, np.bool_))
            or self.num_local < 1
        ):
            raise ValueError(f"num_local must be >= 1; got {self.num_local}")
        if self.max_neighbors == "auto":
            pass
        elif (
            isinstance(self.max_neighbors, (int, np.integer))
            and not isinstance(self.max_neighbors, (bool, np.bool_))
        ):
            if self.max_neighbors < 1:
                raise ValueError(
                    f"max_neighbors must be >= 1; got {self.max_neighbors}"
                )
        else:
            raise ValueError(
                f"max_neighbors must be an integer >= 1 or 'auto'; got {self.max_neighbors!r}"
            )
        if self.cost_evaluation not in {"delta", "brute_force"}:
            raise ValueError(
                f"The 'cost_evaluation' parameter of {self.__class__.__name__} must be a str among "
                f"{{'brute_force', 'delta'}}. Got {self.cost_evaluation!r} instead."
            )

        if (
            not isinstance(self.verbose, (int, np.integer, bool, np.bool_))
            or self.verbose < 0
        ):
            raise ValueError(
                f"The 'verbose' parameter of {self.__class__.__name__} must be an integer >= 0 or a bool. "
                f"Got {self.verbose!r} instead."
            )

        if self.metric_params is not None and not isinstance(self.metric_params, dict):
            raise ValueError(
                f"The 'metric_params' parameter of {self.__class__.__name__} "
                f"must be a dict or None. Got {self.metric_params!r} instead."
            )

        if self.metric == "mahalanobis":
            if not self.metric_params or "VI" not in self.metric_params:
                raise ValueError(
                    "The 'mahalanobis' metric requires the inverse covariance matrix 'VI' "
                    "to be specified in 'metric_params' (e.g. metric_params={'VI': VI})."
                )

        if not callable(self.metric):
            if not isinstance(self.metric, str) or self.metric not in _ALL_VALID_METRICS:
                options_repr = "{" + ", ".join(repr(m) for m in sorted(_ALL_VALID_METRICS)) + "}"
                raise ValueError(
                    f"The 'metric' parameter of {self.__class__.__name__} must be a str among "
                    f"{options_repr} or a callable. Got {self.metric!r} instead."
                )

        allow_nan = (self.metric == "nan_euclidean")
        finite_kw = _finite_kwarg(allow_nan)
        try:
            from sklearn.utils.validation import validate_data

            X = validate_data(
                self, X=X, ensure_min_samples=2, accept_sparse=["csr", "csc"],
                **finite_kw
            )
        except ImportError:
            if hasattr(self, "_validate_data"):
                X = self._validate_data(
                    X, ensure_min_samples=2, accept_sparse=["csr", "csc"],
                    **finite_kw
                )
            else:
                X = check_array(
                    X, ensure_min_samples=2, accept_sparse=["csr", "csc"],
                    **finite_kw
                )
                self.n_features_in_ = X.shape[1]

        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        if self.n_clusters >= n_samples:
            raise ValueError(
                f"n_clusters must be less than n_samples ({n_samples}); got {self.n_clusters}"
            )

        if self.metric == "precomputed":
            if n_samples != n_features:
                raise ValueError(
                    f"Precomputed distance matrix must be square "
                    f"(got shape ({n_samples}, {n_features}))"
                )
            if hasattr(self, "n_features_in_"):
                del self.n_features_in_
            if hasattr(self, "feature_names_in_"):
                del self.feature_names_in_

        return X, random_state, n_samples, n_features

    def _finalize_fit(self, X, best_cost, best_medoids):
        """Set fitted attributes and assign cluster labels."""
        self.inertia_ = float(best_cost)
        self.medoid_indices_ = np.sort(best_medoids)
        self._n_features_out = self.n_clusters

        if self.metric == "precomputed":
            self.cluster_centers_ = None
            dist_to_medoids = X[:, self.medoid_indices_]
            if hasattr(dist_to_medoids, "toarray"):
                dist_to_medoids = dist_to_medoids.toarray()
            self.labels_ = np.argmin(dist_to_medoids, axis=1)
        else:
            self.cluster_centers_ = X[self.medoid_indices_]
            params = self.metric_params if self.metric_params is not None else {}
            if not issparse(X):
                scipy_metric = _SCIPY_METRIC_MAP.get(self.metric, self.metric)
                try:
                    D = cdist(X, self.cluster_centers_, metric=scipy_metric, **params)
                    self.labels_ = np.argmin(D, axis=1)
                except Exception:
                    self.labels_, _ = pairwise_distances_argmin_min(
                        X, self.cluster_centers_, metric=self.metric, metric_kwargs=params
                    )
            else:
                self.labels_, _ = pairwise_distances_argmin_min(
                    X, self.cluster_centers_, metric=self.metric, metric_kwargs=params
                )

        if hasattr(self, "_precomputed_source"):
            del self._precomputed_source

        return self

    def predict(self, X: ArrayLike | "spmatrix") -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            New data to predict. Accepts CSR/CSC sparse matrices.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.

        Raises
        ------
        ValueError
            If the number of features in ``X`` does not match the number of
            features seen during fitting.

        Notes
        -----
        This method uses ``pairwise_distances_argmin_min`` from scikit-learn
        to assign each sample to the nearest medoid.
        """
        check_is_fitted(self)

        if self.metric == "precomputed":
            X = check_array(X, accept_sparse=["csr", "csc"])
            n_train_samples = len(self.labels_)
            if X.shape[1] == n_train_samples:
                dist_to_medoids = X[:, self.medoid_indices_]
            elif X.shape[1] == self.n_clusters:
                dist_to_medoids = X
            else:
                raise ValueError(
                    f"Precomputed X has {X.shape[1]} columns; expected either "
                    f"{n_train_samples} (samples) or {self.n_clusters} (clusters)."
                )
            if hasattr(dist_to_medoids, "toarray"):
                dist_to_medoids = dist_to_medoids.toarray()
            return np.argmin(dist_to_medoids, axis=1)

        allow_nan = (self.metric == "nan_euclidean")
        finite_kw = _finite_kwarg(allow_nan)
        try:
            from sklearn.utils.validation import validate_data

            X = validate_data(
                self, X=X, reset=False, accept_sparse=["csr", "csc"],
                **finite_kw
            )
        except ImportError:
            if hasattr(self, "_validate_data"):
                X = self._validate_data(
                    X, reset=False, accept_sparse=["csr", "csc"],
                    **finite_kw
                )
            else:
                X = check_array(
                    X, accept_sparse=["csr", "csc"],
                    **finite_kw
                )
                if (
                    hasattr(self, "n_features_in_")
                    and X.shape[1] != self.n_features_in_
                ):
                    raise ValueError(
                        f"X has {X.shape[1]} features, but CLARANS is expecting "
                        f"{self.n_features_in_} features as input"
                    )

        params = self.metric_params if self.metric_params is not None else {}
        if not issparse(X):
            scipy_metric = _SCIPY_METRIC_MAP.get(self.metric, self.metric)
            try:
                D = cdist(X, self.cluster_centers_, metric=scipy_metric, **params)
                return np.argmin(D, axis=1)
            except Exception:
                pass

        labels, _ = pairwise_distances_argmin_min(
            X, self.cluster_centers_, metric=self.metric, metric_kwargs=params
        )
        return labels

    def transform(self, X: ArrayLike | "spmatrix") -> np.ndarray:
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        if self.metric == "precomputed":
            X = check_array(X, accept_sparse=["csr", "csc"])
            n_train_samples = len(self.labels_)
            if X.shape[1] == n_train_samples:
                dist_to_medoids = X[:, self.medoid_indices_]
            elif X.shape[1] == self.n_clusters:
                dist_to_medoids = X
            else:
                raise ValueError(
                    f"Precomputed X has {X.shape[1]} columns; expected either "
                    f"{n_train_samples} (samples) or {self.n_clusters} (clusters)."
                )
            return (
                dist_to_medoids.toarray()
                if hasattr(dist_to_medoids, "toarray")
                else np.asarray(dist_to_medoids)
            )

        allow_nan = (self.metric == "nan_euclidean")
        finite_kw = _finite_kwarg(allow_nan)
        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(
                self, X=X, reset=False, accept_sparse=["csr", "csc"],
                **finite_kw
            )
        except ImportError:
            if hasattr(self, "_validate_data"):
                X = self._validate_data(
                    X, reset=False, accept_sparse=["csr", "csc"],
                    **finite_kw
                )
            else:
                X = check_array(
                    X, accept_sparse=["csr", "csc"],
                    **finite_kw
                )

        params = self.metric_params if self.metric_params is not None else {}
        if not issparse(X):
            scipy_metric = _SCIPY_METRIC_MAP.get(self.metric, self.metric)
            try:
                return cdist(X, self.cluster_centers_, metric=scipy_metric, **params)
            except Exception:
                pass

        return pairwise_distances(
            X, self.cluster_centers_, metric=self.metric, **params
        )

    def score(self, X: ArrayLike | "spmatrix", y: Any = None) -> float:
        """Opposite of the total distance of samples to their closest cluster center.

        Enables direct integration with GridSearchCV and cross_val_score.
        Uses transform(X) to evaluate distances to learned medoids on both
        train and unseen test data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            or (n_samples, n_train_samples) if metric='precomputed'.
            New data to score.

        y : Ignored, default=None
            Not used, present here for API consistency with scikit-learn.

        Returns
        -------
        score : float
            Opposite of the total cost (sum of distances to the closest center).
        """
        check_is_fitted(self)
        return -float(np.sum(np.min(self.transform(X), axis=1)))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Get output feature names for transformation.

        The feature names out will be prefixed by the lowercased class name.
        For example, if the transformer outputs 3 features, then the feature names
        out are: `["clarans0", "clarans1", "clarans2"]` (or `fastclarans0`, etc.).

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Only used to validate feature names with the names seen in `fit`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "_n_features_out")
        if input_features is not None and hasattr(self, "feature_names_in_"):
            if len(input_features) != len(self.feature_names_in_):
                raise ValueError(
                    f"input_features should have length equal to the number of "
                    f"features ({len(self.feature_names_in_)}), got {len(input_features)}"
                )
        class_name = self.__class__.__name__.lower()
        return np.asarray(
            [f"{class_name}{i}" for i in range(self._n_features_out)],
            dtype=object,
        )
