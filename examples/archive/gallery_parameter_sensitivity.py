"""Generate `parameter_sensitivity.png` showing cost/runtime for parameter grid.
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS
from clarans.utils import calculate_cost


def main():
    plt.style.use("default")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#2c3e50",
    })
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
    num_locals = [1, 2, 5]
    max_neighbors = [50, 150, 300, 500]

    cost_grid = np.zeros((len(num_locals), len(max_neighbors)))
    time_grid = np.zeros_like(cost_grid)

    for i, nl in enumerate(num_locals):
        for j, mn in enumerate(max_neighbors):
            t0 = time.perf_counter()
            model = CLARANS(n_clusters=4, num_local=nl, max_neighbors=mn, random_state=42)
            model.fit(X)
            time_grid[i, j] = time.perf_counter() - t0
            cost_grid[i, j] = calculate_cost(X, model.medoid_indices_)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200)

    # Cost Heatmap: Blues
    im0 = axes[0].imshow(cost_grid, cmap="Blues", origin="lower", aspect="auto")
    axes[0].set_xticks(range(len(max_neighbors)))
    axes[0].set_xticklabels([str(m) for m in max_neighbors])
    axes[0].set_yticks(range(len(num_locals)))
    axes[0].set_yticklabels([str(n) for n in num_locals])
    axes[0].set_xlabel("max_neighbors")
    axes[0].set_ylabel("num_local")
    axes[0].set_title("(a) Solution Cost (lower is better)", pad=8)
    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.85)
    cbar0.ax.tick_params(labelsize=8)

    cost_min, cost_max = cost_grid.min(), cost_grid.max()
    for i in range(len(num_locals)):
        for j in range(len(max_neighbors)):
            val = cost_grid[i, j]
            norm = (val - cost_min) / (cost_max - cost_min + 1e-8)
            text_color = "white" if norm > 0.65 else "#222222"
            axes[0].text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                color=text_color, fontsize=9
            )

    # Runtime Heatmap: YlOrRd
    im1 = axes[1].imshow(time_grid, cmap="YlOrRd", origin="lower", aspect="auto")
    axes[1].set_xticks(range(len(max_neighbors)))
    axes[1].set_xticklabels([str(m) for m in max_neighbors])
    axes[1].set_yticks(range(len(num_locals)))
    axes[1].set_yticklabels([str(n) for n in num_locals])
    axes[1].set_xlabel("max_neighbors")
    axes[1].set_ylabel("num_local")
    axes[1].set_title("(b) Runtime in seconds", pad=8)
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.85)
    cbar1.ax.tick_params(labelsize=8)

    time_min, time_max = time_grid.min(), time_grid.max()
    for i in range(len(num_locals)):
        for j in range(len(max_neighbors)):
            val = time_grid[i, j]
            norm = (val - time_min) / (time_max - time_min + 1e-8)
            text_color = "white" if norm > 0.65 else "#222222"
            axes[1].text(
                j, i, f"{val:.2f}s",
                ha="center", va="center",
                color=text_color, fontsize=9
            )

    plt.tight_layout()
    out = "parameter_sensitivity.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
