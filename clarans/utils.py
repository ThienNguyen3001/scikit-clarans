import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


def calculate_cost(X, medoid_indices, metric="euclidean"):
    """
    Calculate the total cost (sum of distances) for a given set of medoids.
    """
    medoids = X[medoid_indices]
    _, min_dists = pairwise_distances_argmin_min(X, medoids, metric=metric)
    return np.sum(min_dists)
