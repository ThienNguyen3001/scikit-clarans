"""
==============================================================
Clustering on Asymmetric Precomputed Distance Matrices (D != D^T)
==============================================================

This example demonstrates how CLARANS and FastCLARANS cluster directed networks
and asymmetric dissimilarity matrices (where :math:`D_{ij} \\neq D_{ji}`).

In real-world applications such as urban transportation, delivery logistics,
and directed social networks, travel costs are often asymmetric:

- **Topography**: Biking or driving uphill takes longer than downhill.
- **Traffic systems**: One-way street layouts and tidal traffic flow induce
  directional cost disparities.
- **Directed flows**: Following citations, hyper-links, or supply-chain routes.

scikit-clarans automatically validates distance matrix symmetry via high-speed
Cython kernels and handles asymmetric precomputed distance matrices
without requiring manual symmetrization.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from clarans import FastCLARANS
from clarans.utils import calculate_cost

# 1. Synthesize a 3-community directed travel-time network (N=60 locations)
rng = np.random.RandomState(42)
n_nodes = 60
community_size = 20
n_clusters = 3

# Base symmetric distances within and across communities
D_base = rng.uniform(low=1.0, high=3.0, size=(n_nodes, n_nodes))
for c in range(n_clusters):
    start, end = c * community_size, (c + 1) * community_size
    # Shorter transit times within the same community
    D_base[start:end, start:end] = rng.uniform(0.2, 1.2, size=(community_size, community_size))

# Introduce directional asymmetry: e.g. uphill transit bias or one-way routing
# Directional gradient: traveling from low index to high index is faster than the reverse
directional_bias = np.fromfunction(lambda i, j: 0.08 * np.maximum(0, i - j), (n_nodes, n_nodes))
D_asym = D_base + directional_bias
np.fill_diagonal(D_asym, 0.0)

# Verify asymmetry: ||D - D.T|| > 0
asym_ratio = np.mean(np.abs(D_asym - D_asym.T)) / np.mean(D_asym)
print(f"Mean Directional Asymmetry Ratio: {asym_ratio:.1%}")

# 2. Fit FastCLARANS directly on asymmetric matrix
model = FastCLARANS(
    n_clusters=n_clusters,
    num_local=3,
    metric="precomputed",
    random_state=42,
)
model.fit(D_asym)

# Validate inertia with calculate_cost
exact_cost = calculate_cost(D_asym, model.medoid_indices_, metric="precomputed")
print(f"Optimal Medoid Hubs:           {model.medoid_indices_}")
print(f"FastCLARANS Asymmetric Cost:   {model.inertia_:.2f}")
print(f"Verification Cost Match:       {abs(model.inertia_ - exact_cost) < 1e-9}")

# 3. Visualization: Raw vs. Cluster-Sorted Distance Matrices
sort_idx = np.argsort(model.labels_)
D_sorted = D_asym[sort_idx, :][:, sort_idx]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# Panel (a): Raw Unsorted Asymmetric Distance Matrix
im_a = axes[0].imshow(D_asym, cmap="YlGnBu", origin="upper")
axes[0].set_title(
    f"(a) Raw Asymmetric Distance Matrix\n(Asymmetry: $D_{{ij}} \\neq D_{{ji}}$, {asym_ratio:.1%})",
    fontsize=10.5,
    pad=8,
)
axes[0].set_xlabel("Destination Node ($j$)", fontsize=9.5)
axes[0].set_ylabel("Origin Node ($i$)", fontsize=9.5)
plt.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04, label="Travel Cost")

# Panel (b): Permuted Distance Matrix Ordered by Cluster
im_b = axes[1].imshow(D_sorted, cmap="YlGnBu", origin="upper")
axes[1].set_title(
    f"(b) Clustered Matrix (Ordered by Assigned Hub)\nInertia = {model.inertia_:.1f}",
    fontsize=10.5,
    pad=8,
)
axes[1].set_xlabel("Destination Node (Grouped)", fontsize=9.5)
axes[1].set_ylabel("Origin Node (Grouped)", fontsize=9.5)

# Add partition boundary lines between communities
for k in range(1, n_clusters):
    split = k * community_size - 0.5
    axes[1].axvline(split, color="#d95f02", linestyle="--", linewidth=1.2)
    axes[1].axhline(split, color="#d95f02", linestyle="--", linewidth=1.2)

plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04, label="Travel Cost")

plt.suptitle(
    "FastCLARANS on Directed / Asymmetric Networks (metric='precomputed')",
    fontsize=12,
    y=0.98,
)
plt.tight_layout()
plt.show()
