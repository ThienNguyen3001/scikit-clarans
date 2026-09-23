from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING, Callable, Sequence
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances_argmin_min

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

try:
    from . import _core

    HAS_CYTHON = _core is not None
except ImportError:
    _core = None  # type: ignore[assignment]
    HAS_CYTHON = False

_cython_warning_lock = threading.Lock()
_cython_warning_issued = False


class EfficiencyWarning(UserWarning):
    """Warning issued when falling back to pure Python/NumPy implementation."""


def _warn_cython_unavailable() -> None:
    """Issue an EfficiencyWarning once per session if Cython core is missing."""
    global _cython_warning_issued
    if not HAS_CYTHON and not _cython_warning_issued:
        with _cython_warning_lock:
            if not _cython_warning_issued:
                warnings.warn(
                    "Compiled Cython extensions (_core) are not available; falling back to "
                    "pure Python/NumPy implementation. Performance will be significantly slower. "
                    "To enable C-extension acceleration, compile via `pip install -e .` or "
                    "build from source.",
                    EfficiencyWarning,
                    stacklevel=3,
                )
                _cython_warning_issued = True


__all__ = [
    "calculate_cost",
    "check_medoids",
    "EfficiencyWarning",
    "HAS_CYTHON",
]

_SCIPY_METRIC_MAP = {
    "manhattan": "cityblock",
    "l1": "cityblock",
    "l2": "euclidean",
    "infinity": "chebyshev",
    "sokalmichener": "matching",
    "p": "minkowski",
}


def check_medoids(
    medoids: Sequence[int] | np.ndarray,
    n_samples: int | None = None,
) -> np.ndarray:
    """Validate and convert medoid indices to a 1D NumPy array of integers.

    Parameters
    ----------
    medoids : array-like of shape (n_clusters,)
        Indices representing medoid observations.

    n_samples : int, optional
        Total number of samples in the dataset. If provided, ensures that all
        medoid indices satisfy 0 <= idx < n_samples.

    Returns
    -------
    medoids_arr : np.ndarray of shape (n_clusters,) and dtype np.intp
        Validated 1D array of unique medoid indices.

    Raises
    ------
    ValueError
        If `medoids` is empty, contains duplicate indices, has invalid dimensions,
        contains negative indices, or has indices outside the valid range [0, n_samples - 1].
    TypeError
        If `medoids` contains non-integer elements.
    """
    try:
        medoids_arr = np.asarray(medoids)
    except Exception as exc:
        raise ValueError(f"Could not convert medoids to numpy array: {exc}") from exc

    if medoids_arr.ndim != 1:
        raise ValueError(f"medoid_indices must be 1-dimensional, got shape {medoids_arr.shape}")

    if len(medoids_arr) == 0:
        raise ValueError("medoid_indices cannot be empty.")

    if not np.issubdtype(medoids_arr.dtype, np.integer):
        raise TypeError(f"medoid_indices must contain integers, got dtype {medoids_arr.dtype}")

    medoids_arr = medoids_arr.astype(np.intp)

    if len(np.unique(medoids_arr)) != len(medoids_arr):
        raise ValueError("medoid_indices must not contain duplicate elements.")

    if np.any(medoids_arr < 0):
        raise ValueError(
            f"All medoid indices must be non-negative (>= 0), got min={medoids_arr.min()}."
        )

    if n_samples is not None:
        if np.any(medoids_arr >= n_samples):
            raise ValueError(
                f"All medoid indices must be within [0, {n_samples - 1}], "
                f"got min={medoids_arr.min()}, max={medoids_arr.max()}."
            )

    return medoids_arr


def calculate_cost(
    X: np.ndarray | spmatrix,
    medoid_indices: Sequence[int] | np.ndarray,
    metric: str | Callable = "euclidean",
    metric_params: dict | None = None,
) -> float:
    """Calculate the total cost (sum of distances) for a given set of medoids.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        or shape (n_samples, n_samples) if metric='precomputed'.
        The input samples.

    medoid_indices : array-like of shape (n_clusters,)
        Indices of the medoids in the dataset X.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    Returns
    -------
    cost : float
        The total sum of distances from each point to its nearest medoid.
    """
    medoid_indices = check_medoids(medoid_indices, n_samples=X.shape[0])

    if metric == "precomputed":
        dist_sub = X[:, medoid_indices]
        if hasattr(dist_sub, "toarray"):
            dist_sub = dist_sub.toarray()
        return float(np.sum(np.min(dist_sub, axis=1)))

    medoids = X[medoid_indices]
    params = metric_params if metric_params is not None else {}

    if not issparse(X):
        scipy_metric = (
            _SCIPY_METRIC_MAP.get(metric, metric) if isinstance(metric, str) else metric
        )
        try:
            D = cdist(X, medoids, metric=scipy_metric, **params)
            return float(np.sum(np.min(D, axis=1)))
        except Exception:
            pass

    _, min_dists = pairwise_distances_argmin_min(
        X, medoids, metric=metric, metric_kwargs=params
    )
    return float(np.sum(min_dists))
