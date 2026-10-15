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

X, y_true = make_blobs(n_samples=600, centers=4, cluster_std=0.7, random_state=42)

model = CLARANS(n_clusters=4, num_local=3, init="k-medoids++", random_state=42)
model.fit(X)

# Plot clusters and selected medoids
fig, ax = plt.subplots(figsize=(6.5, 4.8))
colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

for k in range(model.n_clusters):
    mask = model.labels_ == k
    ax.scatter(
        X[mask, 0],
        X[mask, 1],
        c=colors[k],
        s=25,
        alpha=0.65,
        edgecolors="none",
        label=f"Cluster {k + 1}",
    )

ax.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=140,
    linewidths=1.8,
    label="Medoids",
    zorder=10,
)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

ax.set_title(f"CLARANS Clustering (Inertia: {model.inertia_:.1f})", fontsize=11, pad=8)
ax.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=8.5,
)
plt.tight_layout()
plt.show()
