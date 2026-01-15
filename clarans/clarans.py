import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils.validation import (check_array, check_is_fitted,
                                      check_random_state)

from .initialization import (initialize_build, initialize_heuristic,
                             initialize_k_medoids_plus_plus)
from .utils import calculate_cost


class CLARANS(ClusterMixin, BaseEstimator):
    """
    CLARANS (Clustering Large Applications based on RANdomized Search)

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        medoids to generate.

    numlocal : int, default=2
        Number of local minima to find.

    maxneighbor : int, default=None
        Maximum number of neighbors to examine. If None, it is set to
        max(250, 1.25% of k*(n-k)) where k is n_clusters and n is n_samples.

    max_iter : int, default=300
        Maximum number of hops (improvements) allowed in a local search.
        This prevents infinite loops in case the algorithm keeps finding better solutions.

    init : {'random', 'heuristic', 'k-medoids++', 'build', array-like}, default='random'
        Method for initialization:
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        'heuristic': picks the n_clusters points with the smallest sum distance to
        every other point.
        'k-medoids++': selects initial cluster centers for k-medoid clustering
        in a smart way to speed up convergence.
        'build': greedy initialization of the medoids used in the original PAM algorithm.
        Often 'build' is more efficient but slower than other initializations on big datasets,
        and it is also very non-robust to outliers.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    metric : str or callable, default='euclidean'
        Metric used for calculating dissimilarity between observations.
        Supported metrics are those supported by sklearn.metrics.pairwise_distances.

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

    def fit(self, X, y=None):
        """
        Compute CLARANS clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """

        try:
            # Try using validate_data from sklearn.utils.validation (newer sklearn versions)
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X=X, ensure_min_samples=2)
        except ImportError:
            # Fallback to _validate_data (older sklearn versions / internal method)
            if hasattr(self, "_validate_data"):
                X = self._validate_data(X, ensure_min_samples=2)
            else:
                X = check_array(X, ensure_min_samples=2)
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
        best_medoids = None
        self.n_iter_ = 0

        for loc_idx in range(self.numlocal):
            if isinstance(self.init, str) and self.init == "random":
                current_medoids_indices = random_state.choice(
                    n_samples, self.n_clusters, replace=False
                )
            elif isinstance(self.init, str) and self.init == "k-medoids++":
                current_medoids_indices = initialize_k_medoids_plus_plus(
                    X,
                    self.n_clusters,
                    random_state,
                    self.metric
                )
            elif isinstance(self.init, str) and self.init == "heuristic":
                current_medoids_indices = initialize_heuristic(
                    X, self.n_clusters, self.metric
                )
            elif isinstance(self.init, str) and self.init == "build":
                # Implement greedy initialization used in the original PAM algorithm.
                current_medoids_indices = initialize_build(
                    X, self.n_clusters, self.metric
                )
            elif hasattr(self.init, "__array__") or isinstance(self.init, list):
                # User provided array of centers
                init_centers = check_array(self.init)
                if init_centers.shape != (self.n_clusters, n_features):
                    raise ValueError(
                        f"init array must be of shape ({self.n_clusters}, {n_features})"
                    )

                # Find indices of closest points in X to use as medoids
                current_medoids_indices = pairwise_distances_argmin_min(
                    init_centers, X, metric=self.metric
                )[0]

                # Ensure unique
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

            current_cost = calculate_cost(X, current_medoids_indices, self.metric)

            i = 0
            iter_count = 0
            while i < self.maxneighbor_:
                if self.max_iter is not None and iter_count >= self.max_iter:
                    break

                random_medoid_idx = random_state.randint(0, self.n_clusters)

                # Pick a random non-medoid point
                medoid_set = set(current_medoids_indices)
                available_candidates = list(set(range(n_samples)) - medoid_set)
                
                if not available_candidates:
                    # Edge case: all points are medoids
                    break
                    
                random_non_medoid_candidate = random_state.choice(available_candidates)

                neighbor_medoids_indices = current_medoids_indices.copy()
                neighbor_medoids_indices[random_medoid_idx] = (
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

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        try:
            # Try using validate_data from sklearn.utils.validation (newer sklearn versions)
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X=X, reset=False)
        except ImportError:
            # Fallback to _validate_data (older sklearn versions / internal method)
            if hasattr(self, "_validate_data"):
                X = self._validate_data(X, reset=False)
            else:
                X = check_array(X)
                if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
                    raise ValueError(
                        f"X has {X.shape[1]} features, but CLARANS is expecting "
                        f"{self.n_features_in_} features as input"
                    )

        labels, _ = pairwise_distances_argmin_min(
            X, self.cluster_centers_, metric=self.metric
        )
        return labels
