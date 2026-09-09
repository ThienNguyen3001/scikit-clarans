from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


def calculate_cost(
    X: np.ndarray | spmatrix,
    medoid_indices: Sequence[int] | np.ndarray,
    metric: str | Callable = "euclidean",
) -> float:
    """
    Calculate the total cost (sum of distances) for a given set of medoids.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        or shape (n_samples, n_samples) if metric='precomputed'.
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
    if metric == "precomputed":
        dist_sub = X[:, medoid_indices]
        if hasattr(dist_sub, "toarray"):
            dist_sub = dist_sub.toarray()
        return float(np.sum(np.min(dist_sub, axis=1)))

    medoids = X[medoid_indices]
    _, min_dists = pairwise_distances_argmin_min(X, medoids, metric=metric)
    return float(np.sum(min_dists))
