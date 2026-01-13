import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances

class CLARANS(BaseEstimator, ClusterMixin):
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
    def __init__(self, n_clusters=8, numlocal=2, maxneighbor=None, random_state=None):
        self.n_clusters = n_clusters
        self.numlocal = numlocal
        self.maxneighbor = maxneighbor
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
        X = check_array(X)
        random_state = check_random_state(self.random_state)
        n_samples, _ = X.shape

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters should be less than n_samples")

        if self.maxneighbor is None:
             # Heuristic from the original paper/standard implementations
             # specific implementations might vary, but 1.25% of k(n-k) is common reference or a fixed number
             self.maxneighbor_ = max(250, int(0.0125 * self.n_clusters * (n_samples - self.n_clusters)))
        else:
            self.maxneighbor_ = self.maxneighbor

        best_cost = np.inf
        best_medoids = None

        # Distance matrix calculation could be heavy for very large datasets, 
        # but for standard usage we can compute it or compute on fly. 
        # CLARANS is strictly based on medoids, so we deal with indices.
        
        # Optimization: We won't precompute the full distance matrix as CLARANS is designed for large applications
        # and checking neighbors involves swapping one medoid.

        for _ in range(self.numlocal):
            current_medoids_indices = random_state.choice(n_samples, self.n_clusters, replace=False)
            current_medoids_indices.sort() # Sorting helps in consistent state representations
            
            # Calculate initial cost
            current_cost = self._calculate_cost(X, current_medoids_indices)
            
            i = 0
            while i < self.maxneighbor_:
                # Pick a random medoid to replace
                random_medoid_idx = random_state.randint(0, self.n_clusters)
                current_medoid_scanned = current_medoids_indices[random_medoid_idx]
                
                # Pick a random non-medoid point
                # We do this by picking a random index and ensuring it's not in medoids
                # This simple rejection sampling is efficient if k << n
                while True:
                    random_non_medoid_candidate = random_state.randint(0, n_samples)
                    if random_non_medoid_candidate not in current_medoids_indices:
                        break
                
                # Create neighbor node (swapping one medoid)
                neighbor_medoids_indices = current_medoids_indices.copy()
                neighbor_medoids_indices[random_medoid_idx] = random_non_medoid_candidate
                neighbor_medoids_indices.sort()
                
                neighbor_cost = self._calculate_cost(X, neighbor_medoids_indices)
                
                if neighbor_cost < current_cost:
                    current_medoids_indices = neighbor_medoids_indices
                    current_cost = neighbor_cost
                    i = 0 # Reset counter if improvement found
                else:
                    i += 1
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_medoids = current_medoids_indices

        self.medoid_indices_ = best_medoids
        self.cluster_centers_ = X[self.medoid_indices_]
        
        # Assign labels based on final medoids
        # efficient calculation using sklearn utility
        self.labels_, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        
        return self

    def _calculate_cost(self, X, medoid_indices):
        """
        Calculate the total cost (sum of distances) for a given set of medoids.
        """
        # Distances from X to the selected medoids
        medoids = X[medoid_indices]
        # We need the min distance for each point to ANY medoid
        # pairwise_distances_argmin_min returns (min_indices, min_distances)
        _, min_dists = pairwise_distances_argmin_min(X, medoids, metric='euclidean')
        return np.sum(min_dists)

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
        X = check_array(X)
        labels, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return labels

