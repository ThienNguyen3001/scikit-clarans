"""
04_sparse_input.py
==================
Show that CLARANS accepts sparse CSR matrices as input.

Run with: python examples/04_sparse_input.py
"""

from scipy import sparse
from clarans import CLARANS
from sklearn.datasets import make_blobs
import numpy as np


def main():
    X, _ = make_blobs(n_samples=400, centers=3, n_features=6, random_state=0)

    # Make the matrix sparse by zeroing-out small values
    X[np.abs(X) < 1.0] = 0
    X_sparse = sparse.csr_matrix(X)

    model = CLARANS(n_clusters=3, init="k-medoids++", random_state=0)
    model.fit(X_sparse)

    print("Medoid indices:", model.medoid_indices_)
    print("Cluster centers (medoids); shape:", model.cluster_centers_.shape)


if __name__ == "__main__":
    main()
