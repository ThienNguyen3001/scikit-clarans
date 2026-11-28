from __future__ import annotations

import threading
import warnings
from typing import Any, Callable, Sequence
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from sklearn.metrics import DistanceMetric, pairwise_distances

try:
    from . import _core

    HAS_CYTHON = _core is not None
except ImportError:
    _core = None  # type: ignore[assignment]
    HAS_CYTHON = False

_cython_warning_issued = False
_cython_warning_lock = threading.Lock()


class EfficiencyWarning(UserWarning):
    """Warning issued when falling back to pure Python/NumPy implementation."""


def _warn_cython_unavailable() -> None:
    """Issue an EfficiencyWarning once per session if Cython core is missing."""
    global _cython_warning_issued
    if not HAS_CYTHON and not _cython_warning_issued:
        with _cython_warning_lock:
            if not _cython_warning_issued:
                _cython_warning_issued = True
                warnings.warn(
                    "Compiled Cython extensions (_core) are not available; falling back to "
                    "pure Python/NumPy implementation. Performance will be significantly slower. "
                    "To enable C-extension acceleration, compile via `pip install -e .` or "
                    "build from source.",
                    EfficiencyWarning,
                    stacklevel=3,
                )


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


def _is_cdist_unsupported_error(err: Exception) -> bool:
    """Return True if cdist failed due to an unsupported metric rather than invalid args/data."""
    if isinstance(err, ValueError) and "Unknown Distance Metric" in str(err):
        return True
    return not isinstance(err, (ValueError, TypeError))


def _pairwise_fast(
    X: Any,
    Y: Any = None,
    metric: str | Callable = "euclidean",
    metric_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Unified 3-tier distance dispatcher: cdist -> DistanceMetric -> pairwise_distances."""
    params = metric_params if metric_params is not None else {}
    Y_eff = X if Y is None else Y
    if not issparse(X) and not issparse(Y_eff) and isinstance(metric, str):
        mapped = _map_scipy_metric(metric)
        try:
            return cdist(X, Y_eff, metric=mapped, **params)
        except Exception as err:
            if not _is_cdist_unsupported_error(err):
                raise
    if not issparse(X) or isinstance(metric, str):
        try:
            dm = DistanceMetric.get_metric(metric, **params)
            return dm.pairwise(X) if Y is None else dm.pairwise(X, Y_eff)
        except (ValueError, TypeError):
            pass
    return pairwise_distances(X, Y, metric=metric, **params)


def _to_dense(arr: Any) -> np.ndarray:
    """Convert an array or sparse matrix to a dense NumPy array."""
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


def _ensure_2d_input(X: Any) -> Any:
    """Normalize array-like or sparse inputs to 2D ndarrays or CSR/CSC sparse matrices."""
    if issparse(X):
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D sparse matrix, got shape {X.shape}")
        if getattr(X, "format", None) not in ("csr", "csc"):
            return X.tocsr()
        return X
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D array with shape {arr.shape}")
    return arr


def _nan_safe_argmin(D: np.ndarray) -> np.ndarray:
    """Return column index of minimum distance per row, treating NaNs as +inf."""
    if np.issubdtype(D.dtype, np.floating) and np.isnan(D).any():
        return np.argmin(np.where(np.isnan(D), np.inf, D), axis=1)
    return np.argmin(D, axis=1)


def _nan_safe_min(D: np.ndarray) -> np.ndarray:
    """Return minimum distance per row, treating NaNs as +inf."""
    if np.issubdtype(D.dtype, np.floating) and np.isnan(D).any():
        return np.min(np.where(np.isnan(D), np.inf, D), axis=1)
    return np.min(D, axis=1)


def _is_valid_int(val: Any, min_val: int | None = 1, allow_bool: bool = False) -> bool:
    """Check if value is a valid integer >= min_val and optionally allow booleans."""
    if isinstance(val, (bool, np.bool_)):
        return bool(allow_bool and (min_val is None or val >= min_val))
    return bool(isinstance(val, (int, np.integer)) and (min_val is None or val >= min_val))


def _map_scipy_metric(metric: str | Callable) -> str | Callable:
    """Map metric name to SciPy cdist compatible metric identifier."""
    return _SCIPY_METRIC_MAP.get(metric, metric) if isinstance(metric, str) else metric


def check_medoids(
    medoid_indices: Sequence[int] | np.ndarray,
    n_samples: int | None = None,
) -> np.ndarray:
    """Validate and convert medoid indices to a 1D NumPy array of integers.

    Parameters
    ----------
    medoid_indices : array-like of shape (n_clusters,)
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
        If `medoid_indices` is empty, contains duplicate indices, has invalid dimensions,
        contains negative indices, or has indices outside the valid range [0, n_samples - 1].
    TypeError
        If `medoid_indices` contains non-integer elements.
    """
    if medoid_indices is None:
        raise ValueError("medoid_indices cannot be empty.")

    arr = np.asarray(medoid_indices)

    if arr.ndim != 1:
        raise ValueError(f"medoid_indices must be 1-dimensional, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError("medoid_indices cannot be empty.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"medoid_indices must contain integers, got dtype {arr.dtype}")

    arr = arr.astype(np.intp, copy=False)
    if len(np.unique(arr)) != len(arr):
        raise ValueError("medoid_indices must not contain duplicate elements.")
    if np.any(arr < 0):
        raise ValueError(f"All medoid indices must be non-negative (>= 0), got min={arr.min()}.")
    if n_samples is not None and np.any(arr >= n_samples):
        raise ValueError(
            f"All medoid indices must be within [0, {n_samples - 1}], "
            f"got min={arr.min()}, max={arr.max()}."
        )

    return arr


def _resolve_metric_params(
    X: Any, metric: str | Callable, metric_params: dict[str, Any] | None
) -> dict[str, Any]:
    """Return copied metric_params, freezing feature variance V for seuclidean when omitted."""
    params = dict(metric_params) if metric_params is not None else {}
    if metric == "seuclidean" and "V" not in params and X.shape[0] > 1:
        n_rows = X.shape[0]
        if issparse(X):
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
            params["V"] = np.maximum(0.0, mean_sq - mean**2) * (n_rows / (n_rows - 1))
        else:
            params["V"] = np.var(X, axis=0, ddof=1)
    return params


def calculate_cost(
    X: Any,
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
    X_arr = _ensure_2d_input(X)
    medoid_indices = check_medoids(medoid_indices, n_samples=X_arr.shape[0])

    if metric == "precomputed":
        dist_sub = _to_dense(X_arr[:, medoid_indices])
        return float(np.sum(_nan_safe_min(dist_sub), dtype=np.float64))

    medoids = X_arr[medoid_indices]
    params = _resolve_metric_params(X_arr, metric, metric_params)
    D = _pairwise_fast(X_arr, medoids, metric=metric, metric_params=params)
    return float(np.sum(_nan_safe_min(D), dtype=np.float64))


def _nonfinite_swap_components(
    near_dist: np.ndarray,
    second_dist: np.ndarray,
    d_xc: np.ndarray,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Compute exact (inf_delta, finite_delta) terms when deltas contain inf or NaN."""
    d1 = np.where(np.isnan(near_dist), np.inf, near_dist)
    d2 = np.where(np.isnan(second_dist), np.inf, second_dist)
    dc = np.where(np.isnan(d_xc), np.inf, d_xc)

    closer = dc < d1
    d1_inf = np.isinf(d1)

    base_inf = -int(np.count_nonzero(closer & d1_inf))
    with np.errstate(invalid="ignore"):
        fin_closer = np.where(closer, np.where(d1_inf, dc, dc - d1), 0.0)
    base_fin = float(np.sum(fin_closer, dtype=np.float64))

    stay = ~closer
    new_d = np.minimum(dc, d2)
    new_inf = np.isinf(new_d)
    inf_w = np.where(
        stay & d1_inf & ~new_inf,
        -1,
        np.where(stay & ~d1_inf & new_inf, 1, 0),
    )
    with np.errstate(invalid="ignore"):
        fin_w = np.where(
            stay,
            np.where(
                d1_inf & ~new_inf,
                new_d,
                np.where(
                    ~d1_inf & new_inf,
                    -d1,
                    np.where(d1_inf & new_inf, 0.0, new_d - d1),
                ),
            ),
            0.0,
        )
    return base_inf, base_fin, inf_w, fin_w
