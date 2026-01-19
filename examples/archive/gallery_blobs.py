"""Archived: gallery_blobs example (full content copied for history)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS


def main():
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
    model = CLARANS(n_clusters=4, numlocal=5, random_state=42)
    model.fit(X)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=20, cmap="tab10", alpha=0.8)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="x", s=100)
    ax.set_title("CLARANS: blobs")
    ax.set_xticks([])
    ax.set_yticks([])

    out = "gallery_clusters_blobs.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
