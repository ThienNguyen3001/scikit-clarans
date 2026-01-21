"""Archived: gallery_parameter_sensitivity example (full content copied for history)."""

from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS
from clarans.utils import calculate_cost


def main():
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
    numlocals = [1, 2, 5]
    maxneighbors = [50, 200, 500]

    cost_grid = np.zeros((len(numlocals), len(maxneighbors)))
    time_grid = np.zeros_like(cost_grid)

    for i, nl in enumerate(numlocals):
        for j, mn in enumerate(maxneighbors):
            t0 = time.perf_counter()
            model = CLARANS(n_clusters=4, numlocal=nl, maxneighbor=mn, random_state=42)
            model.fit(X)
            time_grid[i, j] = time.perf_counter() - t0
            cost_grid[i, j] = calculate_cost(X, model.medoid_indices_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axes[0].imshow(cost_grid, cmap="viridis", origin="lower")
    axes[0].set_xticks(range(len(maxneighbors)))
    axes[0].set_xticklabels([str(m) for m in maxneighbors])
    axes[0].set_yticks(range(len(numlocals)))
    axes[0].set_yticklabels([str(n) for n in numlocals])
    axes[0].set_xlabel("maxneighbor")
    axes[0].set_ylabel("numlocal")
    axes[0].set_title("Final cost")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(time_grid, cmap="magma", origin="lower")
    axes[1].set_xticks(range(len(maxneighbors)))
    axes[1].set_xticklabels([str(m) for m in maxneighbors])
    axes[1].set_yticks(range(len(numlocals)))
    axes[1].set_yticklabels([str(n) for n in numlocals])
    axes[1].set_xlabel("maxneighbor")
    axes[1].set_ylabel("numlocal")
    axes[1].set_title("Runtime (s)")
    fig.colorbar(im1, ax=axes[1])

    out = "parameter_sensitivity.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
