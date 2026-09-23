"""
============================================================
Distance Metrics and Metric Parameters in CLARANS Clustering
============================================================

This example demonstrates how different distance metrics and their hyperparameters
via ``metric_params`` (Euclidean, Manhattan, Minkowski with :math:`p=3`, and
Mahalanobis with covariance inverse :math:`V^{-1}`) affect cluster partition
boundaries and medoid placements in CLARANS.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from clarans import CLARANS
from clarans.utils import calculate_cost


def main():
    X, _ = make_blobs(n_samples=400, centers=3, n_features=2, random_state=42)

    # Precompute inverse covariance matrix for Mahalanobis metric
    VI = np.linalg.pinv(np.cov(X.T))

    configs = [
        ("Euclidean", "euclidean", None),
        ("Manhattan (L1)", "manhattan", None),
        ("Minkowski (p=3)", "minkowski", {"p": 3}),
        ("Mahalanobis", "mahalanobis", {"VI": VI}),
    ]

    fig, axes = plt.subplots(1, len(configs), figsize=(16, 4))

    for ax, (title, metric, metric_params) in zip(axes, configs):
        model = CLARANS(
            n_clusters=3,
            num_local=3,
            metric=metric,
            metric_params=metric_params,
            random_state=42,
        )
        model.fit(X)
        cost = calculate_cost(
            X, model.medoid_indices_, metric=metric, metric_params=metric_params
        )

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
        param_str = f"\nmetric_params={metric_params}" if metric_params else ""
        ax.set_title(f"{title}\nCost: {cost:.1f}{param_str}", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    plt.suptitle("CLARANS with Different Distance Metrics & metric_params", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
