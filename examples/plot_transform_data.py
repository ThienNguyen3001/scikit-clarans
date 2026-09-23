"""
=================================================
Feature Transformation via Cluster-Distance Space
=================================================

This example demonstrates using CLARANS as a feature transformer (`transform`),
mapping sample coordinates into distance-to-medoid representations.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS


def main():
    X, _ = make_blobs(n_samples=250, centers=3, n_features=4, random_state=42)

    model = CLARANS(n_clusters=3, random_state=42)
    model.fit(X)

    # Transform data to cluster-distance space: shape (n_samples, n_clusters)
    X_trans = model.transform(X)
    print(f"Original shape:    {X.shape}")
    print(f"Transformed shape: {X_trans.shape}")

    # Plot sample distances to Medoid 0 vs Medoid 1
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(
        X_trans[:, 0],
        X_trans[:, 1],
        c=model.labels_,
        cmap="tab10",
        s=25,
        alpha=0.7,
    )
    plt.xlabel("Distance to Medoid 0")
    plt.ylabel("Distance to Medoid 1")
    plt.title("Transformed Feature Space: Distance to Medoids")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
