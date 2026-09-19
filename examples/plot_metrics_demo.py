"""
================================================
Effect of Distance Metrics on CLARANS Clustering
================================================

This example demonstrates how different distance metrics (Euclidean, Manhattan,
Cosine) affect cluster partition boundaries and medoid placements in CLARANS.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS
from clarans.utils import calculate_cost


def main():
    X, _ = make_blobs(n_samples=400, centers=3, n_features=2, random_state=42)
    metrics = ["euclidean", "manhattan", "cosine"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))

    for ax, metric in zip(axes, metrics):
        model = CLARANS(n_clusters=3, num_local=3, metric=metric, random_state=42)
        model.fit(X)
        cost = calculate_cost(X, model.medoid_indices_, metric=metric)

        ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="tab10", s=20, alpha=0.7)
        ax.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            c="black",
            marker="X",
            s=120,
            linewidths=1.8,
            label="Medoids",
        )
        ax.set_title(f"{metric.capitalize()} (Cost: {cost:.1f})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    plt.suptitle("CLARANS with Different Distance Metrics", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
