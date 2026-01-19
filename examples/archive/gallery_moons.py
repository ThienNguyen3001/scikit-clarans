"""Generate `gallery_clusters_moons.png`.

Standalone example that demonstrates CLARANS on two interleaving moons.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from clarans import CLARANS


def main():
    X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
    model = CLARANS(n_clusters=2, numlocal=5, random_state=42)
    model.fit(X)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=20, cmap="tab10", alpha=0.8)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="x", s=100)
    ax.set_title("CLARANS: moons")
    ax.set_xticks([])
    ax.set_yticks([])

    out = "gallery_clusters_moons.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
