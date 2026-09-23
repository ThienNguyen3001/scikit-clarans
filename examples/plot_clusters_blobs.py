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

COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]


def main():
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
    model = CLARANS(n_clusters=4, num_local=5, random_state=42)
    model.fit(X)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for k in range(4):
        mask = model.labels_ == k
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[k], s=25, alpha=0.65, edgecolors="none")

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
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
