"""Archived: gallery_runtime example (full content copied for history)."""

from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS, FastCLARANS
from sklearn.cluster import KMeans


def main():
    Ns = [200, 500, 1000, 2000, 5000, 10000]
    clarans_times = []
    fast_times = []
    kmeans_times = []

    for N in Ns:
        X, _ = make_blobs(n_samples=N, centers=5, random_state=42)

        t0 = time.perf_counter()
        CLARANS(n_clusters=5, random_state=42).fit(X)
        clarans_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        FastCLARANS(n_clusters=5, random_state=42).fit(X)
        fast_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        KMeans(n_clusters=4, random_state=42).fit(X)
        kmeans_times.append(time.perf_counter() - t0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ns, clarans_times, marker="o", label="CLARANS")
    ax.plot(Ns, fast_times, marker="o", label="FastCLARANS")
    ax.plot(Ns, kmeans_times, marker="o", label="KMeans")
    ax.set_xlabel("n samples")
    ax.set_ylabel("time (s)")
    ax.set_title("Runtime scaling")
    ax.legend()

    out = "runtime_scaling.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")

    print(kmeans_times, clarans_times, fast_times)


if __name__ == "__main__":
    main()
