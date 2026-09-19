"""
==========================================================
Side-by-Side Comparison: CLARANS, FastCLARANS, and K-Means
==========================================================

This example performs a direct visual and performance comparison between
classic CLARANS, FastCLARANS, and scikit-learn's standard KMeans clusterer.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from clarans import CLARANS, FastCLARANS
from clarans.utils import calculate_cost

COLORS = ["#2b5c8f", "#d95f02", "#7570b3"]


def main():
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=0.70, random_state=42)
    models = [
        ("CLARANS", CLARANS(n_clusters=3, num_local=5, random_state=42), "x", "Medoid"),
        ("FastCLARANS", FastCLARANS(n_clusters=3, num_local=5, random_state=42), "x", "Medoid"),
        ("K-Means", KMeans(n_clusters=3, random_state=42, n_init=10), "+", "Centroid"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    for ax, (name, model, marker, label_name) in zip(axes, models):
        t0 = time.perf_counter()
        model.fit(X)
        elapsed = time.perf_counter() - t0

        for k in range(3):
            mask = model.labels_ == k
            ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[k], s=18, alpha=0.65, edgecolors="none")

        centers = getattr(model, "cluster_centers_", None)
        if centers is not None:
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                marker=marker,
                s=85,
                c="black",
                linewidths=1.8,
                label=label_name,
                zorder=5,
            )

        if hasattr(model, "medoid_indices_"):
            cost = calculate_cost(X, model.medoid_indices_)
            stat_str = f"Time: {elapsed:.3f}s | Cost: {cost:.1f}"
        else:
            stat_str = f"Time: {elapsed:.3f}s | Inertia: {model.inertia_:.1f}"

        ax.set_title(name, fontsize=11, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.8)

        ax.text(
            0.05,
            0.05,
            stat_str,
            transform=ax.transAxes,
            fontsize=8.5,
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.85,
            ),
        )
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
