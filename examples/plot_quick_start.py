"""
===================================
CLARANS 2D Clustering Demonstration
===================================

This example demonstrates how to use the CLARANS algorithm on a 2D synthetic
blobs dataset, displaying the discovered clusters and actual dataset medoids.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS

# 1. Generate synthetic dataset
X, y_true = make_blobs(n_samples=600, centers=4, cluster_std=0.7, random_state=42)

# 2. Fit CLARANS
model = CLARANS(n_clusters=4, num_local=3, init="k-medoids++", random_state=42)
model.fit(X)

# 3. Plot clusters and selected medoids
plt.figure(figsize=(7, 5))
colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

for k in range(model.n_clusters):
    mask = model.labels_ == k
    plt.scatter(
        X[mask, 0],
        X[mask, 1],
        c=colors[k],
        s=25,
        alpha=0.7,
        label=f"Cluster {k}",
    )

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=150,
    linewidths=2,
    label="Medoids",
    zorder=10,
)

plt.title(f"CLARANS Clustering (Inertia: {model.inertia_:.2f})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
