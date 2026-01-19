"""Archived: gallery_silhouette example (full content copied for history)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from clarans import CLARANS, FastCLARANS


def main():
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
    ks = range(2, 9)
    methods = {
        "CLARANS": lambda k: CLARANS(n_clusters=k, numlocal=3, random_state=42),
        "FastCLARANS": lambda k: FastCLARANS(n_clusters=k, numlocal=3, random_state=42),
        "KMeans": lambda k: KMeans(n_clusters=k, random_state=42),
    }

    results = {name: [] for name in methods}

    for k in ks:
        for name, factory in methods.items():
            model = factory(k)
            model.fit(X)
            labels = model.labels_
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
            else:
                score = float("nan")
            results[name].append(score)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, scores in results.items():
        ax.plot(list(ks), scores, marker="o", label=name)
    ax.set_xlabel("k (n_clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score vs k")
    ax.legend()

    out = "silhouette_vs_k.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
