"""
========================================
CLARANS 2D Clustering on Synthetic Blobs
========================================

This example demonstrates how CLARANS identifies clusters and true medoids
(actual data points) on a 2D Gaussian blobs dataset.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS

plt.rcParams.update({"axes.edgecolor": "#cccccc", "axes.linewidth": 0.8})

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
model = CLARANS(n_clusters=4, num_local=5, random_state=42)
model.fit(X)

fig, ax = plt.subplots(figsize=(6.0, 4.5))
ax.scatter(
    X[:, 0],
    X[:, 1],
    c=model.labels_,
    cmap="tab10",
    s=25,
    alpha=0.65,
    edgecolors="none",
)

centers = model.cluster_centers_
ax.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="x",
    s=90,
    c="black",
    linewidths=1.8,
    label="Medoids",
)

ax.set_title("CLARANS on Synthetic Blobs ($k=4$)", fontsize=11, pad=8)
ax.set_xticks([])
ax.set_yticks([])

ax.legend(loc="upper right", frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
