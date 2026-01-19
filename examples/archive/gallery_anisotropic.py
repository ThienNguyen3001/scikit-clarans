"""Generate `gallery_clusters_anisotropic.png`.

Standalone example that demonstrates CLARANS on an anisotropic dataset.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS


def main():
    X, _ = make_blobs(n_samples=500, centers=3, random_state=170)
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = X.dot(transformation)

    model = CLARANS(n_clusters=3, numlocal=5, random_state=42)
    model.fit(X)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=20, cmap="tab10", alpha=0.8)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="x", s=100)
    ax.set_title("CLARANS: anisotropic")
    ax.set_xticks([])
    ax.set_yticks([])

    out = "gallery_clusters_anisotropic.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
