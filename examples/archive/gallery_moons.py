"""Generate `gallery_clusters_moons.png`.

Minimalist example for moons dataset.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from clarans import CLARANS

COLORS = ["#2b5c8f", "#d95f02"]


def main():
    plt.style.use("default")
    X, _ = make_moons(n_samples=500, noise=0.06, random_state=42)
    model = CLARANS(n_clusters=2, num_local=5, random_state=42)
    model.fit(X)

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=200)
    for k in range(2):
        mask = model.labels_ == k
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[k], s=20, alpha=0.65, edgecolors="none")

    centers = model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=90, c="black", linewidths=1.8, label="Medoids")

    ax.set_title("CLARANS on Interleaving Moons ($k=2$)", fontsize=11, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout()

    out = "gallery_clusters_moons.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
