from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

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
    CLARANS (Clustering Large Applications based on RANdomized Search)

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form (also the number of medoids to
        generate).

    numlocal : int, default=2
        The number of local searches to perform.
        CLARANS runs the search process ``numlocal`` times starting from
        different random nodes to reduce the chance of getting stuck in
        poor local minima. Increasing this improves solution quality but
        increases runtime.

    maxneighbor : int, default=None
        The maximum number of neighbors (random swaps) to examine during
        each step. If ``None``, it defaults to ``max(250, 1.25% of k*(n-k))``.
        Higher values make the algorithm behave more like PAM (checking
        more neighbors); lower values make it faster but more random.

    max_iter : int, default=300
        The maximum number of successful swaps (improvements) allowed per
        local search. This acts as a safeguard against infinite loops.

    init : {'random', 'heuristic', 'k-medoids++', 'build', array-like}, default='random'
        Strategy for selecting initial medoids:

        - ``'random'``: Selects ``n_clusters`` random points. Fast but can
          result in poor starting points.
        - ``'heuristic'``: Selects points that are "central" to the data
          (minimizing distance to all others).
        - ``'k-medoids++'``: Optimized probabilistic initialization (similar
          to k-means++) for faster convergence.
        - ``'build'``: The greedy initialization from the original PAM
          algorithm. High quality but slow (O(N^2)).

    metric : str or callable, default='euclidean'
        The distance metric to use. Supports all metrics from
        ``sklearn.metrics.pairwise_distances`` (e.g., 'euclidean',
        'manhattan', 'cosine').

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers (medoids).

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    medoid_indices_ : ndarray of shape (n_clusters,)
        Indices of the medoids in the training set X.

    Notes
    -----
    - Time complexity: each local search evaluates up to ``maxneighbor``
      candidate swaps, and each cost evaluation is O(n * k) (distance
      to medoids), so the worst-case runtime is roughly
      O(numlocal * maxneighbor * n * k).
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

    def __init__(
        self,
        n_clusters=8,
        numlocal=2,
        maxneighbor=None,
        max_iter=300,
        init="random",
        metric="euclidean",
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.numlocal = numlocal
        self.maxneighbor = maxneighbor
        self.max_iter = max_iter
        self.init = init
        self.metric = metric
        self.random_state = random_state

    def __sklearn_tags__(self):
        """Declare estimator capabilities for scikit-learn's check_estimator.

        CLARANS can accept CSR/CSC sparse matrices as input.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def fit(self, X: ArrayLike | "spmatrix", y: Any = None) -> "CLARANS":
        """
        Compute CLARANS clustering.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training instances to cluster. Accepts CSR/CSC sparse matrices.

        y : Ignored
            Not used, present here for API consistency.

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
        - Time complexity: each local search evaluates up to ``maxneighbor``
          candidate swaps, and each cost evaluation is O(n * k) (distance
          to medoids), so the worst-case runtime is roughly
          O(numlocal * maxneighbor * n * k).
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
            raise ValueError("n_clusters must be less than n_samples")

        if self.maxneighbor is None:
            self.maxneighbor_ = max(
                250, int(0.0125 * self.n_clusters * (n_samples - self.n_clusters))
            )
        else:
            self.maxneighbor_ = self.maxneighbor

        best_cost = np.inf
        best_medoids = np.empty(self.n_clusters, dtype=int)
        self.n_iter_ = 0

        all_indices = np.arange(n_samples)

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

            current_medoids_indices = np.array(current_medoids_indices, dtype=int)

            current_cost = calculate_cost(X, current_medoids_indices, self.metric)

            i = 0
            iter_count = 0

            while i < self.maxneighbor_:
                if self.max_iter is not None and iter_count >= self.max_iter:
                    break

                random_medoid_pos = random_state.randint(0, self.n_clusters)

                mask = np.ones(n_samples, dtype=bool)
                mask[current_medoids_indices] = False
                available_candidates = np.flatnonzero(mask)

                if available_candidates.size == 0:
                    break

                random_non_medoid_candidate = random_state.choice(available_candidates)

                neighbor_medoids_indices = current_medoids_indices.copy()
                neighbor_medoids_indices[random_medoid_pos] = (
                    random_non_medoid_candidate
                )

                neighbor_cost = calculate_cost(X, neighbor_medoids_indices, self.metric)

                if neighbor_cost < current_cost:
                    current_medoids_indices = neighbor_medoids_indices
                    current_cost = neighbor_cost
                    i = 0
                    iter_count += 1
                else:
                    i += 1

            self.n_iter_ += max(1, iter_count)

            if current_cost < best_cost:
                best_cost = current_cost
                best_medoids = current_medoids_indices

        self.medoid_indices_ = best_medoids
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

        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X=X, reset=False, accept_sparse=["csr", "csc"])
        except ImportError:
            if hasattr(self, "_validate_data"):
                X = self._validate_data(X, reset=False, accept_sparse=["csr", "csc"])
            else:
                X = check_array(X, accept_sparse=["csr", "csc"])

        return pairwise_distances(X, self.cluster_centers_, metric=self.metric)
