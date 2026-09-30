r"""
==================================================================
Monitoring Search Progress and Restart Convergence Trajectories
==================================================================

This example demonstrates search progress monitoring and restart convergence
using the ``verbose`` parameter and fitted estimator attributes.

CLARANS performs randomized local searches across ``num_local`` restarts to
explore different basins of attraction. At each restart, candidate medoid swaps
are evaluated until ``max_neighbors_`` consecutive non-improving samples are
tested (convergence).

Setting ``verbose=1`` prints real-time iteration statistics, while fitted
attributes (``n_iter_``, ``n_swaps_``, ``max_neighbors_``, ``n_neighbors_``)
record cumulative evaluation and swap metrics.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from clarans import FastCLARANS

# Generate synthetic dataset
n_samples = 1200
n_clusters = 5
X, y_true = make_blobs(
    n_samples=n_samples,
    centers=n_clusters,
    cluster_std=0.9,
    random_state=42,
)

# 1. Fit with verbose=1 to demonstrate console progress logging
print("Fitting FastCLARANS with verbose=1:")
model = FastCLARANS(
    n_clusters=n_clusters,
    num_local=4,
    verbose=1,
    random_state=42,
)
model.fit(X)

print("\nFitted search attributes:")
print(f"  Best inertia_     : {model.inertia_:.4f}")
print(f"  Total n_iter_     : {model.n_iter_} candidate evaluations")
print(f"  Total n_swaps_    : {model.n_swaps_} accepted medoid swaps")
print(f"  max_neighbors_    : {model.max_neighbors_} (auto threshold)")
print(f"  n_neighbors_      : {model.n_neighbors_} (non-medoid candidate pool)")

# 2. Analyze search trajectory across different num_local restart counts
restarts = [1, 2, 3, 4, 5, 6]
inertias = []
total_evals = []
total_swaps = []

for r in restarts:
    m = FastCLARANS(
        n_clusters=n_clusters,
        num_local=r,
        random_state=42,
    )
    m.fit(X)
    inertias.append(m.inertia_)
    total_evals.append(m.n_iter_)
    total_swaps.append(m.n_swaps_)

# 3. Visualize search trajectory and final clustering
fig = plt.figure(figsize=(12, 4.5))

# Panel 1: Inertia improvement over restarts
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(restarts, inertias, marker="o", color="#4c72b0", linewidth=2)
best_r = restarts[int(np.argmin(inertias))]
best_cost = min(inertias)
ax1.scatter([best_r], [best_cost], color="#c44e52", s=100, zorder=5, label=f"Best (r={best_r})")
ax1.set_xlabel("Number of restarts (num_local)")
ax1.set_ylabel("Final inertia (Total Cost)")
ax1.set_title("Solution Quality vs. Restarts")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# Panel 2: Cumulative evaluations and swaps
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(
    restarts, total_evals, marker="s", color="#55a868", linewidth=2, label="Evals (n_iter_)",
)
ax2_twin = ax2.twinx()
ax2_twin.plot(
    restarts, total_swaps, marker="^", color="#8172b3", linewidth=2, linestyle="--",
    label="Swaps (n_swaps_)",
)
ax2.set_xlabel("Number of restarts (num_local)")
ax2.set_ylabel("Total candidate evaluations", color="#55a868")
ax2_twin.set_ylabel("Total accepted swaps", color="#8172b3")
ax2.set_title("Search Effort Tracking")
ax2.grid(True, linestyle="--", alpha=0.5)

# Panel 3: Final clustering result
ax3 = fig.add_subplot(1, 3, 3)
scatter = ax3.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="tab10", alpha=0.5, s=15)
medoids = model.cluster_centers_
ax3.scatter(
    medoids[:, 0],
    medoids[:, 1],
    c="black",
    marker="X",
    s=120,
    edgecolors="white",
    linewidths=1.5,
    label="Medoids",
)
ax3.set_title(f"Fitted Clusters (inertia={model.inertia_:.1f})")
ax3.legend(loc="upper right")
ax3.set_xticks([])
ax3.set_yticks([])

plt.suptitle("FastCLARANS Search Monitoring and Restart Trajectories", fontsize=13)
plt.tight_layout()
plt.show()
