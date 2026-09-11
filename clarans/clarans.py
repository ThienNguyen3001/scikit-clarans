from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from .initialization import (
    initialize_build,
    initialize_heuristic,
    initialize_k_medoids_plus_plus,
)
from .utils import calculate_cost

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


class CLARANS(ClusterMixin, TransformerMixin, BaseEstimator):
    """
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

    init : {'k-medoids++', 'random', 'heuristic', 'build', array-like}, default='k-medoids++'
        Strategy for selecting initial medoids:

        - ``'k-medoids++'``: Optimized probabilistic initialization (similar
          to k-means++) for faster convergence.
        - ``'random'``: Selects ``n_clusters`` random points. Fast but can
          result in poor starting points.
        - ``'heuristic'``: Selects points that are "central" to the data
          (minimizing distance to all others).
        - ``'build'``: The greedy initialization from the original PAM
          algorithm. High quality but slow (O(N^2)).

    metric : str or callable, default='euclidean'
        The distance metric to use. Supports all metrics from
        ``sklearn.metrics.pairwise_distances`` (e.g., 'euclidean',
        'manhattan', 'cosine', 'precomputed').

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for medoid swaps and random
        initialization. Pass an int for reproducible output across multiple
        function calls.

    cache : bool, default=True
        Whether to use distance caching (nearest and second-nearest medoid
        distances d1, d2) to accelerate candidate swap evaluations in O(n*d)
        instead of recalculating the full clustering cost from scratch in O(n*k*d).
        If False, runs the classic brute-force cost recalculation at each candidate
        evaluation.

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
    Ng, R. T., & Han, J. (2002). CLARANS: A method for clustering objects for spatial data mining. 
    IEEE transactions on knowledge and data engineering, 14(5), 1003-1016.

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
        random_state=None,
        cache=True,
    ):
        self.n_clusters = n_clusters
        self.num_local = num_local
        self.max_neighbors = max_neighbors
        self.init = init
        self.metric = metric
        self.random_state = random_state
        self.cache = cache

    def _prepare_initial_medoids(self, X, random_state):
        """Pre-compute initial medoids if the initialization strategy is deterministic.

        For deterministic strategies ('build', 'heuristic', or explicit array),
        computing initial medoids once before the ``num_local`` loop avoids costly
        O(N^2) recalculations across local search restarts.
        """
        is_deterministic = (
            (isinstance(self.init, str) and self.init in ("build", "heuristic"))
            or hasattr(self.init, "__array__")
            or isinstance(self.init, list)
        )
        if is_deterministic:
            if self.num_local > 1:
                if isinstance(self.init, str):
                    warnings.warn(
                        f"The '{self.init}' initialization is deterministic. Running "
                        f"multiple local searches (num_local={self.num_local}) "
                        f"will start from the exact same initial medoids. Consider using "
                        f"num_local=1 or 'k-medoids++' for diverse restarts.",
                        UserWarning,
                    )
                else:
                    warnings.warn(
                        f"An explicit init array was provided. Running multiple local "
                        f"searches (num_local={self.num_local}) will start from "
                        f"the exact same initial medoids.",
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
            if init_centers.shape != (self.n_clusters, n_features):
                raise ValueError(
                    f"init array must be of shape ({self.n_clusters}, {n_features})"
                )

            current_medoids_indices, _ = pairwise_distances_argmin_min(
                init_centers, X, metric=self.metric
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
        else:
            raise ValueError(f"Unknown init method: {self.init}")

        return np.array(current_medoids_indices, dtype=int)

    def __sklearn_tags__(self):
        """Declare estimator capabilities for scikit-learn's check_estimator.

        CLARANS can accept CSR/CSC sparse matrices as input.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        if self.metric == "precomputed":
            tags.input_tags.pairwise = True
        return tags

    def _more_tags(self):
        return {"pairwise": self.metric == "precomputed"}

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
        X, random_state, n_samples, n_features = self._validate_input_and_params(X)

        if self.max_neighbors == "auto":
            self.max_neighbors_ = max(
                250, int(0.0125 * self.n_clusters * (n_samples - self.n_clusters))
            )
        else:
            self.max_neighbors_ = int(self.max_neighbors)

        best_cost = np.inf
        best_medoids = np.empty(self.n_clusters, dtype=int)
        best_n_iter = 0
        best_n_swaps = 0

        deterministic_medoids = self._prepare_initial_medoids(X, random_state)

        for loc_idx in range(self.num_local):
            if deterministic_medoids is not None:
                current_medoids_indices = deterministic_medoids.copy()
            else:
                current_medoids_indices = self._initialize_medoids(X, random_state)

            if self.cache:
                near_idx_map, near_dist, second_dist = self._update_cache(
                    X, current_medoids_indices
                )
                current_cost = float(np.sum(near_dist))
            else:
                current_cost = calculate_cost(X, current_medoids_indices, self.metric)

            i = 0
            swap_count = 0
            eval_count = 0

            while i < self.max_neighbors_:
                eval_count += 1
                random_medoid_pos = random_state.randint(0, self.n_clusters)

                mask = np.ones(n_samples, dtype=bool)
                mask[current_medoids_indices] = False
                available_candidates = np.flatnonzero(mask)

                if available_candidates.size == 0:
                    break

                random_non_medoid_candidate = random_state.choice(available_candidates)

                if self.cache:
                    cand_row = X[
                        random_non_medoid_candidate : random_non_medoid_candidate + 1
                    ]
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
                        total_delta = candidate_cost - current_cost
                    else:
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

                    if total_delta < 0:
                        current_medoids_indices[random_medoid_pos] = (
                            random_non_medoid_candidate
                        )
                        near_idx_map, near_dist, second_dist = self._update_cache(
                            X, current_medoids_indices
                        )
                        current_cost = float(np.sum(near_dist))
                        i = 0
                        swap_count += 1
                    else:
                        i += 1
                else:
                    neighbor_medoids_indices = current_medoids_indices.copy()
                    neighbor_medoids_indices[random_medoid_pos] = (
                        random_non_medoid_candidate
                    )

                    neighbor_cost = calculate_cost(
                        X, neighbor_medoids_indices, self.metric
                    )

                    if neighbor_cost < current_cost:
                        current_medoids_indices = neighbor_medoids_indices
                        current_cost = neighbor_cost
                        i = 0
                        swap_count += 1
                    else:
                        i += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_medoids = current_medoids_indices.copy()
                best_n_iter = eval_count
                best_n_swaps = swap_count

        self.n_iter_ = best_n_iter
        self.n_swaps_ = best_n_swaps

        return self._finalize_fit(X, best_cost, best_medoids)

    def _update_cache(
        self, X: np.ndarray | "spmatrix", medoids_indices: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute nearest and second-nearest medoid information on-the-fly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        medoids_indices : array-like of shape (n_clusters,)
            Indices of the current medoids.

        Returns
        -------
        near_idx_map : ndarray of shape (n_samples,)
            Index (0..k-1) of the nearest medoid for each sample.
        near_dist : ndarray of shape (n_samples,)
            Distance from each sample to its nearest medoid.
        second_dist : ndarray of shape (n_samples,)
            Distance from each sample to its second nearest medoid.
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

    def _validate_input_and_params(self, X):
        """Validate estimator parameters and input data array."""
        if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1; got {self.n_clusters}")
        if (
            not isinstance(self.num_local, (int, np.integer))
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
        if not isinstance(self.cache, (bool, np.bool_)):
            raise ValueError(f"cache must be a boolean; got {self.cache}")

        try:
            from sklearn.utils.validation import validate_data

            X = validate_data(
                self, X=X, ensure_min_samples=2, accept_sparse=["csr", "csc"]
            )
        except ImportError:
            if hasattr(self, "_validate_data"):
                X = self._validate_data(
                    X, ensure_min_samples=2, accept_sparse=["csr", "csc"]
                )
            else:
                X = check_array(X, ensure_min_samples=2, accept_sparse=["csr", "csc"])
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
            self.labels_, _ = pairwise_distances_argmin_min(
                X, self.cluster_centers_, metric=self.metric
            )

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

        try:
            from sklearn.utils.validation import validate_data

            X = validate_data(self, X=X, reset=False, accept_sparse=["csr", "csc"])
        except ImportError:
            if hasattr(self, "_validate_data"):
                X = self._validate_data(X, reset=False, accept_sparse=["csr", "csc"])
            else:
                X = check_array(X, accept_sparse=["csr", "csc"])
                if (
                    hasattr(self, "n_features_in_")
                    and X.shape[1] != self.n_features_in_
                ):
                    raise ValueError(
                        f"X has {X.shape[1]} features, but CLARANS is expecting "
                        f"{self.n_features_in_} features as input"
                    )

        labels, _ = pairwise_distances_argmin_min(
            X, self.cluster_centers_, metric=self.metric
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

        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X=X, reset=False, accept_sparse=["csr", "csc"])
        except ImportError:
            if hasattr(self, "_validate_data"):
                X = self._validate_data(X, reset=False, accept_sparse=["csr", "csc"])
            else:
                X = check_array(X, accept_sparse=["csr", "csc"])

        return pairwise_distances(X, self.cluster_centers_, metric=self.metric)

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
        try:
            from sklearn.utils.validation import _generate_get_feature_names_out

            return _generate_get_feature_names_out(
                self, self._n_features_out, input_features=input_features
            )
        except ImportError:
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
