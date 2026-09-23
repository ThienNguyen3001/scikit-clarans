"""
===========================================
Clustering Sparse CSR Matrices with CLARANS
===========================================

This example demonstrates how CLARANS efficiently clusters high-dimensional
sparse data stored as SciPy CSR/CSC sparse matrices.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS


def main():
    # Generate data and make it sparse by zeroing out low values
    X, _ = make_blobs(n_samples=500, centers=3, n_features=20, random_state=42)
    X[np.abs(X) < 1.5] = 0.0
    X_sparse = sparse.csr_matrix(X)
    sparsity = 100.0 * (1.0 - X_sparse.nnz / (X.shape[0] * X.shape[1]))

    print(f"Dataset shape: {X_sparse.shape}")
    print(f"Sparsity: {sparsity:.1f}%")

    model = CLARANS(n_clusters=3, num_local=3, random_state=42)
    model.fit(X_sparse)

    print(f"Selected medoid indices: {model.medoid_indices_}")
    print(f"Inertia (Total Distance Cost): {model.inertia_:.2f}")

    # Visualize cluster sizes
    plt.figure(figsize=(6, 4))
    counts = np.bincount(model.labels_)
    plt.bar(range(len(counts)), counts, color="#2b5c8f", width=0.5)
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Samples")
    plt.title(f"CLARANS on Sparse CSR Matrix ({sparsity:.1f}% Sparse)")
    plt.xticks(range(len(counts)))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
