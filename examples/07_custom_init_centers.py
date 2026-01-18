"""
07_custom_init_centers.py
=========================
Demonstrate passing a custom array-like `init` to CLARANS.

Run: python examples/07_custom_init_centers.py
"""

import numpy as np
from clarans import CLARANS


def main():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 2)

    # Create two custom centers (shape must be (n_clusters, n_features))
    # We'll intentionally pass a duplicate to show how the algorithm handles it.
    custom_init = np.array([[0.0, 0.0], [0.0, 0.0]])

    model = CLARANS(n_clusters=2, init=custom_init, random_state=0)
    model.fit(X)

    print("Medoid indices:", model.medoid_indices_)
    print("Cluster centers:", model.cluster_centers_)


if __name__ == "__main__":
    main()
