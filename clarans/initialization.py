import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min


def initialize_heuristic(X, n_clusters, metric="euclidean"):
    """
    Picks the n_clusters points with the smallest sum distance to every other point.
    """
    # This requires O(N^2) complexity
    D = pairwise_distances(X, metric=metric)
    dist_sums = np.sum(D, axis=1)
    current_medoids_indices = np.argsort(dist_sums)[:n_clusters]
    return current_medoids_indices


def initialize_build(X, n_clusters, metric="euclidean"):
    """
    Implement PAM BUILD initialization.
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


def initialize_k_medoids_plus_plus(X, n_clusters, random_state, metric="euclidean"):
    """
    Initialize medoids using k-medoids++.
    Refactored to be vectorized and avoid redundant distance calculations.
    """
    n_samples, _ = X.shape
    medoid_indices = []

    first_medoid = random_state.randint(0, n_samples)
    medoid_indices.append(first_medoid)

    dists = pairwise_distances(X, X[first_medoid].reshape(1, -1), metric=metric).flatten()
    min_dists = dists

    for _ in range(1, n_clusters):
        d2 = min_dists**2
        total_d2 = np.sum(d2)

        if total_d2 == 0:
            candidates = list(set(range(n_samples)) - set(medoid_indices))
            if not candidates:
                break
            next_medoid = random_state.choice(candidates)
        else:
            probs = d2 / total_d2
            next_medoid = random_state.choice(n_samples, p=probs)

        medoid_indices.append(next_medoid)

        new_dists = pairwise_distances(X, X[next_medoid].reshape(1, -1), metric=metric).flatten()
        min_dists = np.minimum(min_dists, new_dists)

    return np.array(medoid_indices)
