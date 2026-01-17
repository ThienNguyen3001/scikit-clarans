import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


def calculate_cost(X, medoid_indices, metric="euclidean"):
    """
    Calculate the total cost (sum of distances) for a given set of medoids.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    medoid_indices : array-like of shape (n_clusters,)
        Indices of the medoids in the dataset X.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances.

    Returns
    -------
    cost : float
        The total sum of distances from each point to its nearest medoid.
    """
    medoids = X[medoid_indices]
    _, min_dists = pairwise_distances_argmin_min(X, medoids, metric=metric)
    return np.sum(min_dists)
