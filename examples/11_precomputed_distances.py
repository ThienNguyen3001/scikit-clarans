"""Example 11: Clustering with Precomputed Distance Matrices.

This example demonstrates how to use CLARANS and FastCLARANS with
precomputed distance matrices (metric="precomputed"), allowing clustering
on non-Euclidean, graph, string, or custom pairwise distance data.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from clarans import CLARANS, FastCLARANS


def main():
    print("=" * 60)
    print("Example 11: Precomputed Distance Matrix Clustering")
    print("=" * 60)

    # 1. Generate synthetic data and compute custom pairwise distance matrix
    X, _ = make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
    D = pairwise_distances(X, metric="manhattan")
    print(f"Computed Manhattan distance matrix of shape: {D.shape}")

    # 2. Fit CLARANS with metric='precomputed'
    clarans = CLARANS(
        n_clusters=3,
        num_local=3,
        max_neighbors=30,
        metric="precomputed",
        random_state=42,
    )
    clarans.fit(D)

    print("\n--- CLARANS Results ---")
    print(f"Selected Medoid Indices: {clarans.medoid_indices_}")
    print(f"Total Inertia (Cost):    {clarans.inertia_:.4f}")
    print(f"Cluster Sizes:           {np.bincount(clarans.labels_)}")

    # 3. Fit FastCLARANS with metric='precomputed'
    fast_clarans = FastCLARANS(
        n_clusters=3,
        num_local=3,
        max_neighbors=30,
        metric="precomputed",
        random_state=42,
    )
    fast_clarans.fit(D)

    print("\n--- FastCLARANS Results ---")
    print(f"Selected Medoid Indices: {fast_clarans.medoid_indices_}")
    print(f"Total Inertia (Cost):    {fast_clarans.inertia_:.4f}")
    print(f"Cluster Sizes:           {np.bincount(fast_clarans.labels_)}")

    # 4. Predict cluster labels for query distances
    # Can pass query distance to all training samples: shape (n_query, n_samples)
    query_distances = D[:5]
    predicted = fast_clarans.predict(query_distances)
    print(f"\nPredicted labels for first 5 samples: {predicted}")
    print(f"Actual fitted labels:                 {fast_clarans.labels_[:5]}")


if __name__ == "__main__":
    main()
