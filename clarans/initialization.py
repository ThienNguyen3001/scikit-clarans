import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min


def initialize_heuristic(X, n_clusters, metric="euclidean"):
    """
    Picks the n_clusters points with the smallest sum distance to every other point.
    """
    # This requires O(N^2) complexity to compute all pairwise distances
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
    medoids.append(first_medoid)

    for _ in range(1, n_clusters):
        best_candidate = -1
        best_gain = -np.inf

        if len(medoids) > 0:
            dist_to_nearest = D[:, medoids].min(axis=1)
        else:
            dist_to_nearest = np.full(n_samples, np.inf)

        candidates = list(set(range(n_samples)) - set(medoids))


        for cand in candidates:
            d_cand = D[:, cand]
            gain = np.sum(np.maximum(dist_to_nearest - d_cand, 0))

            if gain > best_gain:
                best_gain = gain
                best_candidate = cand

        medoids.append(best_candidate)

    return np.array(medoids)


def initialize_k_medoids_plus_plus(X, n_clusters, random_state, metric="euclidean"):
    """
    Initialize medoids using k-medoids++.
    """
    n_samples, _ = X.shape
    medoid_indices = []

    first_medoid = random_state.randint(0, n_samples)
    medoid_indices.append(first_medoid)

    for _ in range(1, n_clusters):
        current_medoids = X[medoid_indices]
        _, min_dists = pairwise_distances_argmin_min(
            X, current_medoids, metric=metric
        )

        d2 = min_dists**2
        total_d2 = np.sum(d2)

        if total_d2 == 0:
            candidates = list(set(range(n_samples)) - set(medoid_indices))
            next_medoid = random_state.choice(candidates)
        else:
            probs = d2 / total_d2
            next_medoid = random_state.choice(n_samples, p=probs)

        medoid_indices.append(next_medoid)

    return np.array(medoid_indices)
