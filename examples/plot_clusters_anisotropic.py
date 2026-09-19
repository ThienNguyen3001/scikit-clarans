"""
===============================================
CLARANS on Anisotropically Distributed Clusters
===============================================

This example demonstrates how CLARANS identifies medoids on elongated,
anisotropically stretched clusters generated via linear transformation.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS

COLORS = ["#2b5c8f", "#d95f02", "#7570b3"]


def main():
    X, _ = make_blobs(n_samples=500, centers=3, random_state=170)
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = X.dot(transformation)

    model = CLARANS(n_clusters=3, num_local=5, random_state=42)
    model.fit(X)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for k in range(3):
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

    ax.set_title("CLARANS on Anisotropic Clusters ($k=3$)", fontsize=11, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)

    ax.legend(loc="upper left", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
