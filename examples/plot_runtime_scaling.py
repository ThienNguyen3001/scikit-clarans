"""
==============================================
Runtime Scaling vs. Dataset Size (num_local=2)
==============================================

This example plots empirical execution times across increasing dataset sizes N
for CLARANS, FastCLARANS, and K-Means.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from clarans import CLARANS, FastCLARANS


def main():
    Ns = [400, 800, 1500, 3000]
    clarans_times = []
    fast_times = []
    kmeans_times = []

    for N in Ns:
        X, _ = make_blobs(n_samples=N, centers=4, cluster_std=0.60, random_state=42)

        tc, tf, tk = 0.0, 0.0, 0.0
        n_repeats = 2
        for seed in range(42, 42 + n_repeats):
            t0 = time.perf_counter()
            CLARANS(n_clusters=4, num_local=2, random_state=seed).fit(X)
            tc += time.perf_counter() - t0

            t0 = time.perf_counter()
            FastCLARANS(n_clusters=4, num_local=2, random_state=seed).fit(X)
            tf += time.perf_counter() - t0

            t0 = time.perf_counter()
            KMeans(n_clusters=4, random_state=seed, n_init=10).fit(X)
            tk += time.perf_counter() - t0

        clarans_times.append(tc / n_repeats)
        fast_times.append(tf / n_repeats)
        kmeans_times.append(tk / n_repeats)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#e5e5e5")
    ax.xaxis.grid(False)

    ax.plot(
        Ns,
        clarans_times,
        marker="o",
        markersize=5,
        linewidth=1.6,
        color="#2b5c8f",
        label="CLARANS",
    )
    ax.plot(
        Ns,
        fast_times,
        marker="s",
        markersize=5,
        linewidth=1.6,
        color="#1b9e77",
        label="FastCLARANS",
    )
    ax.plot(
        Ns,
        kmeans_times,
        marker="^",
        markersize=5,
        linewidth=1.6,
        color="#d95f02",
        label="K-Means",
    )

    ax.set_xlabel("Number of samples ($N$)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime Scaling vs. Dataset Size (num_local=2)", pad=10)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
