"""
==============================================
Predicting Cluster Labels on Unseen Query Data
==============================================

This example fits CLARANS on training data and visualizes how new, unseen
samples are assigned to their nearest fitted cluster medoids.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS


def main():
    X_train, _ = make_blobs(n_samples=400, centers=3, n_features=2, random_state=42)
    model = CLARANS(n_clusters=3, init="k-medoids++", random_state=42)
    model.fit(X_train)

    # Generate new unseen query points
    rng = np.random.RandomState(99)
    X_query = rng.uniform(
        low=[X_train[:, 0].min(), X_train[:, 1].min()],
        high=[X_train[:, 0].max(), X_train[:, 1].max()],
        size=(40, 2),
    )
    query_preds = model.predict(X_query)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    # Plot training data
    for k in range(3):
        mask = model.labels_ == k
        ax.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            c=colors[k],
            s=20,
            alpha=0.45,
            label=f"Train Cluster {k}",
        )

    # Plot query samples with colored boundaries
    for k in range(3):
        mask = query_preds == k
        ax.scatter(
            X_query[mask, 0],
            X_query[mask, 1],
            c=colors[k],
            marker="s",
            s=70,
            edgecolors="black",
            linewidths=1.2,
            label=f"Predicted Query (Cluster {k})",
        )

    # Plot Medoids
    ax.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=160,
        c="black",
        linewidths=2,
        label="Medoids",
        zorder=10,
    )

    ax.set_title("CLARANS Cluster Predictions for New Unseen Samples")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
