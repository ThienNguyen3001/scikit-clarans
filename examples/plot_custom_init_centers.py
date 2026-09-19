"""
===================================================
User-Provided Initial Candidate Centers for CLARANS
===================================================

This example demonstrates how to pass explicit domain-specific coordinates
as the initial medoids (`init=array_like`), which CLARANS automatically snaps
to the nearest actual samples in the training dataset.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS


def main():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

    # User defines 3 approximate initial seeds (e.g. from domain knowledge)
    user_seeds = np.array([[-5.0, -5.0], [0.0, 5.0], [5.0, -2.0]])

    model = CLARANS(n_clusters=3, init=user_seeds, num_local=1, random_state=42)
    model.fit(X)

    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="tab10", s=20, alpha=0.5)

    # Plot user-provided candidate coordinates
    plt.scatter(
        user_seeds[:, 0],
        user_seeds[:, 1],
        c="red",
        marker="^",
        s=120,
        linewidths=1.5,
        edgecolors="black",
        label="User Input Seeds",
    )

    # Plot final converged medoids
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        c="black",
        marker="X",
        s=150,
        linewidths=2,
        label="Final Fitted Medoids",
        zorder=10,
    )

    plt.title("CLARANS with User-Provided Initial Seeds")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
