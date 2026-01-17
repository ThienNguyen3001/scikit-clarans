import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state


def initialize_heuristic(X, n_clusters, metric="euclidean"):
    """
    Initialize medoids using a heuristic approach.
    
    Picks the n_clusters points with the smallest sum distance to every other point.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    n_clusters : int
        The number of clusters to form.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.

    Returns
    -------
    current_medoids_indices : ndarray of shape (n_clusters,), dtype int
        Indices of the selected medoids in the dataset.

    Notes
    -----
    This method computes the full pairwise distance matrix and therefore has
    O(n^2) time and memory complexity.
    """
    # This requires O(N^2) complexity
    D = pairwise_distances(X, metric=metric)
    dist_sums = np.sum(D, axis=1)
    current_medoids_indices = np.argpartition(dist_sums, n_clusters - 1)[:n_clusters]
    return current_medoids_indices


def initialize_build(X, n_clusters, metric="euclidean"):
    """
    Initialize medoids using the PAM BUILD step.
    
    Greedily selects the first medoid that minimizes total distance, then
    subsequently adds medoids that maximally decrease the total cost.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    n_clusters : int
        The number of clusters to form.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.

    Returns
    -------
    medoids : ndarray of shape (n_clusters,), dtype int
        Indices of the selected medoids in the dataset.
    """
    n_samples = X.shape[0]
    medoids = []

    # Calculate full distance matrix potentially (expensive) or compute on fly
    # Using euclidean_distances or pairwise_distances
    D = pairwise_distances(X, metric=metric)

    dist_sums = D.sum(axis=1)
    first_medoid = np.argmin(dist_sums)
    medoids.append(int(first_medoid))

    dist_to_nearest = D[:, first_medoid]

    for _ in range(1, n_clusters):

        # Mask for medoids
        is_medoid = np.zeros(n_samples, dtype=bool)
        is_medoid[medoids] = True

        # We only care about candidates
        candidate_indices = np.where(~is_medoid)[0]

        D_candidates = D[:, candidate_indices]
        diffs = dist_to_nearest[:, np.newaxis] - D_candidates
        gains = np.sum(np.maximum(diffs, 0), axis=0)

        best_candidate_idx_in_candidates = np.argmax(gains)
        best_candidate = candidate_indices[best_candidate_idx_in_candidates]

        medoids.append(int(best_candidate))

        dist_to_nearest = np.minimum(dist_to_nearest, D[:, best_candidate])

    return np.array(medoids)


def initialize_k_medoids_plus_plus(X, 
                                   n_clusters, 
                                   random_state=None, 
                                   metric="euclidean", 
                                   n_local_trials=None):
    """
    Initialize medoids using k-medoids++ (similar to k-means++).

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    n_clusters : int
        The number of clusters to form.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        If None, the function uses a small logarithmic default.

    Returns
    -------
    medoids : ndarray of shape (n_clusters,), dtype int
        Indices of the selected medoids in the dataset.

    Notes
    -----
    This implementation follows the k-means++ style seeding but uses distances
    squared and picks medoids (data indices) rather than centroids.
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]
    medoid_indices = np.empty(n_clusters, dtype=int)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    first_medoid = random_state.randint(0, n_samples)
    medoid_indices[0] = first_medoid

    closest = pairwise_distances(
        X,
        X[first_medoid].reshape(1, -1),
        metric=metric,
    ).flatten()

    closest_dist_sq = closest ** 2
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot

        cumsum_dist = np.cumsum(closest_dist_sq)

        candidate_ids = np.searchsorted(cumsum_dist, rand_vals)
        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        # Compute distances from candidates to all points
        candidates_X = X[candidate_ids]

        dists_candidates = pairwise_distances(candidates_X, X, metric=metric) ** 2

        best_candidate = None
        best_pot = None
        best_dist_sq = None

        for i in range(n_local_trials):
            new_dist_sq = np.minimum(closest_dist_sq, dists_candidates[i])
            new_pot = new_dist_sq.sum()

            if best_candidate is None or new_pot < best_pot:
                best_candidate = candidate_ids[i]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        medoid_indices[c] = best_candidate
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return medoid_indices
