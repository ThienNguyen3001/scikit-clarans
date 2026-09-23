"""
=============================================
Clustering with Precomputed Distance Matrices
=============================================

This example demonstrates how to use CLARANS and FastCLARANS with
precomputed pairwise distance matrices (`metric='precomputed'`), enabling
clustering on graph, geodesic, or custom non-Euclidean distance data.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from clarans import CLARANS, FastCLARANS


def main():
    X, _ = make_blobs(n_samples=180, centers=3, n_features=4, random_state=42)
    # Generate Manhattan distance matrix
    D = pairwise_distances(X, metric="manhattan")

    # Fit CLARANS with precomputed distances
    model_clarans = CLARANS(n_clusters=3, num_local=3, metric="precomputed", random_state=42)
    model_clarans.fit(D)

    # Fit FastCLARANS with precomputed distances
    model_fast = FastCLARANS(n_clusters=3, num_local=3, metric="precomputed", random_state=42)
    model_fast.fit(D)

    print(
        f"CLARANS Medoids:     {model_clarans.medoid_indices_} | "
        f"Cost: {model_clarans.inertia_:.2f}"
    )
    print(
        f"FastCLARANS Medoids: {model_fast.medoid_indices_} | "
        f"Cost: {model_fast.inertia_:.2f}"
    )

    # Plot pairwise distance heatmap sorted by cluster labels
    sort_idx = model_clarans.labels_.argsort()
    D_sorted = D[sort_idx, :][:, sort_idx]

    plt.figure(figsize=(6, 5))
    plt.imshow(D_sorted, cmap="viridis", origin="lower")
    plt.colorbar(label="Manhattan Distance")
    plt.title("Sorted Precomputed Distance Matrix by Clusters")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
