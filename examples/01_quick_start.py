"""
01_quick_start.py
=================
A compact, runnable example showing a simple CLARANS workflow:

- Generate 2D blob data
- Fit CLARANS
- Print medoid indices and labels
- Plot the resulting clusters and medoids

Run this script with: python examples/01_quick_start.py
"""

from clarans import CLARANS
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def main():
    X, _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)

    model = CLARANS(n_clusters=4, numlocal=3, init="k-medoids++", random_state=42)
    model.fit(X)

    print("Medoid Indices:", model.medoid_indices_)
    print("First 10 Labels:", model.labels_[:10])

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="tab10", s=20, alpha=0.7)
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        c="black",
        marker="*",
        s=200,
        label="Medoids",
    )
    plt.title("CLARANS quick start: clusters and medoids")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
