"""Type stubs for Cython C-extension module clarans._core."""

from typing import Tuple
import numpy as np

def clarans_delta(
    near_idx_map: np.ndarray,
    near_dist: np.ndarray,
    second_dist: np.ndarray,
    d_xc: np.ndarray,
    random_medoid_pos: int,
    n_samples: int,
) -> float: ...

def fastpam1_delta(
    near_idx_map: np.ndarray,
    near_dist: np.ndarray,
    second_dist: np.ndarray,
    d_xc: np.ndarray,
    n_samples: int,
    n_clusters: int,
    delta_buf: np.ndarray | None = None,
) -> Tuple[int, float, np.ndarray]: ...

def update_cache_2min(
    subD: np.ndarray,
    n_samples: int,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def pam_build_step(
    D: np.ndarray,
    candidate_indices: np.ndarray,
    dist_to_nearest: np.ndarray,
    n_samples: int,
    n_candidates: int,
) -> Tuple[int, float]: ...

def kmedoids_pp_trials(
    closest_dist_sq: np.ndarray,
    dists_candidates: np.ndarray,
    candidate_ids: np.ndarray,
    current_medoids: np.ndarray,
    n_samples: int,
    n_local_trials: int,
    n_current_medoids: int,
) -> Tuple[int, float, np.ndarray]: ...

def is_matrix_symmetric(
    D: np.ndarray,
    n_samples: int,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool: ...

