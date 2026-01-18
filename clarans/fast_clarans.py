import numpy as np
import warnings
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.utils.validation import check_random_state, check_array
from clarans.clarans import CLARANS 
from clarans.initialization import *

class FastCLARANS(CLARANS):
    """
    FastCLARANS: A variant of CLARANS using delta cost computation logic 
    from FastPAM for faster O(k) runtime.
    
    References
    ----------
    Schubert, E., & Rousseeuw, P. J. (2021). Fast and eager k-medoids clustering: 
    O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. 
    Information Systems, 101, 101804.
    """

    def fit(self, X, y=None):
        """
        Perform FastCLARANS clustering.
        """
        X = check_array(X, accept_sparse=["csr", "csc"])
        self.n_features_in_ = X.shape[1]
        random_state = check_random_state(self.random_state)
        n_samples, _ = X.shape

        if self.n_clusters >= n_samples:
            raise ValueError("n_clusters must be less than n_samples")

        # default maxneighbor if not specified
        if self.maxneighbor is None:
            self.maxneighbor_ = max(
                250, int(0.0125 * self.n_clusters * (n_samples - self.n_clusters))
            )
        else:
            self.maxneighbor_ = self.maxneighbor

        best_cost = np.inf
        best_medoids = None
        self.n_iter_ = 0

        # precompute full distance matrix (may be memory intensive)
        D = pairwise_distances(X, metric=self.metric)

        # --- Local Search loop (numlocal) ---
        for loc_idx in range(self.numlocal):
            # 1. Initialize medoids (supports same options as CLARANS)
            if isinstance(self.init, str) and self.init == "random":
                current_medoids_indices = random_state.choice(
                    n_samples, self.n_clusters, replace=False
                )
            elif isinstance(self.init, str) and self.init == "k-medoids++":
                current_medoids_indices = initialize_k_medoids_plus_plus(
                    X,
                    self.n_clusters,
                    random_state,
                    self.metric,
                )
            elif isinstance(self.init, str) and self.init == "heuristic":
                current_medoids_indices = initialize_heuristic(
                    X, self.n_clusters, self.metric
                )
            elif isinstance(self.init, str) and self.init == "build":
                current_medoids_indices = initialize_build(
                    X, self.n_clusters, self.metric
                )
            elif hasattr(self.init, "__array__") or isinstance(self.init, list):
                init_centers = check_array(self.init)
                if init_centers.shape != (self.n_clusters, self.n_features_in_):
                    raise ValueError(
                        f"init array must be of shape ({self.n_clusters}, {self.n_features_in_})"
                    )

                current_medoids_indices = pairwise_distances_argmin_min(
                    init_centers, X, metric=self.metric
                )[0]

                if len(set(current_medoids_indices)) < self.n_clusters:
                    warnings.warn(
                        "Provided init centers map to duplicate points in X. "
                        "Filling duplicates with random points."
                    )
                    current_medoids_indices = list(set(current_medoids_indices))
                    remaining = self.n_clusters - len(current_medoids_indices)
                    available = list(set(range(n_samples)) - set(current_medoids_indices))
                    if len(available) < remaining:
                        raise ValueError(
                            "Not enough unique points to fill up to n_clusters."
                        )
                    current_medoids_indices.extend(
                        random_state.choice(available, remaining, replace=False)
                    )
                    current_medoids_indices = np.array(current_medoids_indices)
            else:
                raise ValueError(f"Unknown init method: {self.init}")
            
            # Ensure medoids are a sorted numpy array for consistent indexing
            current_medoids_indices.sort()

            # Compute nearest and second-nearest medoid information for all points
            # Returns: near_idx_map (index in current_medoids_indices), near_dist, second_dist
            near_idx_map, near_dist, second_dist = self._update_cache(n_samples, current_medoids_indices, D)
            current_cost = np.sum(near_dist)

            i = 0
            iter_count = 0
            
            # --- Neighbor loop (maxneighbor) ---
            # Sample non-medoid candidates and evaluate swap deltas
            while i < self.maxneighbor_:
                if self.max_iter is not None and iter_count >= self.max_iter:
                    break

                # Select a random non-medoid candidate
                while True:
                    candidate_idx = random_state.randint(0, n_samples)
                    if candidate_idx not in current_medoids_indices:
                        break

                # Distances from candidate to all points
                d_xc = D[candidate_idx]

                # FastPAM-style delta computations
                # removal_loss: cost increase if a medoid is removed
                removal_loss = np.zeros(self.n_clusters)
                diff = second_dist - near_dist
                removal_loss += np.bincount(near_idx_map, weights=diff, minlength=self.n_clusters)

                # Benefit if candidate becomes nearest for some points
                mask_better_than_nearest = d_xc < near_dist
                delta_td_plus_xc = np.sum(d_xc[mask_better_than_nearest] - near_dist[mask_better_than_nearest])

                # total_delta initialized with removal_loss + global benefit
                total_delta = removal_loss + delta_td_plus_xc

                # Adjust per-medoid contributions (two cases)
                mask_better_than_second = d_xc < second_dist

                term1 = near_dist[mask_better_than_nearest] - second_dist[mask_better_than_nearest]
                idx1 = near_idx_map[mask_better_than_nearest]
                total_delta += np.bincount(idx1, weights=term1, minlength=self.n_clusters)

                mask_case2 = (~mask_better_than_nearest) & mask_better_than_second
                term2 = d_xc[mask_case2] - second_dist[mask_case2]
                idx2 = near_idx_map[mask_case2]
                total_delta += np.bincount(idx2, weights=term2, minlength=self.n_clusters)

                # 5. Find the best medoid to swap
                min_delta_idx = np.argmin(total_delta)
                min_delta = total_delta[min_delta_idx]

                if min_delta < 0:
                    # Found improvement! Swap!
                    # Swap medoid at position min_delta_idx with candidate
                    current_medoids_indices[min_delta_idx] = candidate_idx
                    current_medoids_indices.sort() # Sort again for consistency
                    
                    current_cost += min_delta
                    
                    # Update cache (must update entire cache after swap)
                    # Although O(n*k), swaps are less frequent than checks.
                    near_idx_map, near_dist, second_dist = self._update_cache(n_samples, current_medoids_indices, D)
                    
                    i = 0 # Reset counter as in original CLARANS
                    iter_count += 1
                else:
                    i += 1

            self.n_iter_ += max(1, iter_count)

            if current_cost < best_cost:
                best_cost = current_cost
                best_medoids = current_medoids_indices.copy()

        # End, assign results
        self.medoid_indices_ = best_medoids
        self.cluster_centers_ = X[self.medoid_indices_]
        
        # Final label assignment
        self.labels_, _ = pairwise_distances_argmin_min(
            X, self.cluster_centers_, metric=self.metric
        )
        
        return self

    def _update_cache(self, n_samples, medoids_indices, D):
        """
        Helper to compute nearest and second-nearest medoid distances for all points.

        Returns:
            near_idx_map: (n_samples,) indices in medoids_indices (0..k-1)
            near_dist: (n_samples,) distance to nearest medoid
            second_dist: (n_samples,) distance to second nearest medoid
        """
        # Extract distances to current medoids (submatrix)
        subD = D[:, medoids_indices]

        # Find two smallest distances per row efficiently using argpartition
        if self.n_clusters >= 2:
            partitioned_idx = np.argpartition(subD, 1, axis=1)
            smallest_idx = partitioned_idx[:, 0]
            second_smallest_idx = partitioned_idx[:, 1]

            near_dist = subD[np.arange(n_samples), smallest_idx]
            second_dist = subD[np.arange(n_samples), second_smallest_idx]
            near_idx_map = smallest_idx
        else:
            # k = 1 case
            near_dist = subD[:, 0]
            second_dist = np.full(n_samples, np.inf)
            near_idx_map = np.zeros(n_samples, dtype=int)

        return near_idx_map, near_dist, second_dist