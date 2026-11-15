"""
============================================================
Benchmarking K-Medoids: PAM, CLARA, CLARANS, and FastCLARANS
============================================================

This example provides an empirical benchmark across four landmark algorithms
in the Partitioning Around Medoids (k-medoids) family:

1. **PAM** (Partitioning Around Medoids, Kaufman & Rousseeuw, 1990):
   Evaluates all :math:`k(n - k)` pairwise swaps in an exhaustive greedy search.
   Finds high-quality medoids but scales with quadratic complexity :math:`\\mathcal{O}(k(n-k)^2)`.
   Available in ``scikit-learn-extra`` via :class:`sklearn_extra.cluster.KMedoids` with
   ``method='pam'``.
2. **CLARA** (Clustering Large Applications, Kaufman & Rousseeuw, 1990):
   Draws multiple independent sub-samples of size :math:`40 + 2k`, executes PAM
   on each sub-sample, and retains the medoids minimizing total dissimilarity across
   the entire dataset. Fast, but may miss high-quality medoids if cluster boundaries
   are not well-represented in the sampled subsets.
   Available in ``scikit-learn-extra`` via :class:`sklearn_extra.cluster.CLARA`.
3. **CLARANS** (Ng & Han, 2002):
   Graph-based randomized local search that samples neighboring medoid sets
   up to a budget of ``max_neighbors``, escaping local optima via multiple restarts
   (``num_local``). Scalable to large datasets without full distance matrix allocations.
4. **FastCLARANS** (Schubert & Rousseeuw, 2021):
   Combines randomized graph exploration with vectorized FastPAM1 delta calculations,
   evaluating swaps across all :math:`k` medoids simultaneously in a single pass.
   Achieves clustering quality on par with PAM while running up to two orders of
   magnitude faster.

.. note::

    **Installing scikit-learn-extra on modern Python / NumPy 2.x:**
    If you are running in an environment with NumPy 2.x (standard for Python 3.12+),
    install ``scikit-learn-extra`` directly from its development branch where NumPy 2.x
    compatibility has been merged:

    .. code-block:: bash

        pip install git+https://github.com/scikit-learn-contrib/scikit-learn-extra.git

    The current release on PyPI (``0.3.0``) predates NumPy 2.0 and does not provide
    pre-built binary wheels for Python 3.12 or 3.13.

.. note::

    **Why use logarithmic runtime scaling and relative inertia?**
    Because PAM evaluates swaps exhaustively, its execution time on medium datasets
    dwarfs stochastic and sampled approaches. On a standard linear scale, PAM creates
    a massive vertical bar that flattens FastCLARANS and CLARA to near zero, making them
    visually indistinguishable.
    To provide a clear, readable comparison:

    * **Runtime** is plotted on a **logarithmic scale** (:math:`\\log_{10}`), allowing
      speedup ratios (:math:`10\\times` to :math:`100\\times`) to be clearly read.
    * **Clustering Quality** is plotted as **Relative Inertia (%)** normalized against
      the PAM baseline (:math:`100\\%`), making trade-offs transparent without
      exaggerating small differences.
    * **Features** are scaled using :class:`~sklearn.preprocessing.StandardScaler` so that
      distances across coordinate dimensions remain balanced and unbiased.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import CLARA, KMedoids

from clarans import CLARANS, FastCLARANS

plt.rcParams.update({"axes.edgecolor": "#cccccc", "axes.linewidth": 0.8})

# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------
n_samples = 3000
n_clusters = 5
random_state = 42

X, _ = make_blobs(
    n_samples=n_samples,
    centers=n_clusters,
    cluster_std=0.75,
    random_state=random_state,
)

# Standardize coordinates to remove dimensional scale bias
X = StandardScaler().fit_transform(X)

models = [
    (
        "PAM",
        KMedoids(
            n_clusters=n_clusters,
            method="pam",
            init="build",
            random_state=random_state,
        ),
        "#7570b3",
    ),
    (
        "CLARA",
        CLARA(
            n_clusters=n_clusters,
            n_sampling_iter=5,
            random_state=random_state,
        ),
        "#e7298a",
    ),
    (
        "CLARANS",
        CLARANS(
            n_clusters=n_clusters,
            num_local=2,
            random_state=random_state,
        ),
        "#2b5c8f",
    ),
    (
        "FastCLARANS",
        FastCLARANS(
            n_clusters=n_clusters,
            num_local=2,
            random_state=random_state,
        ),
        "#1b9e77",
    ),
]

results = []
print("Evaluating K-Medoids algorithms...")
for label, model, color in models:
    t0 = time.perf_counter()
    model.fit(X)
    elapsed = time.perf_counter() - t0
    results.append(
        {
            "label": label,
            "name": label.split("\n")[0],
            "model": model,
            "time": elapsed,
            "inertia": model.inertia_,
            "color": color,
        }
    )

pam_time = results[0]["time"]
pam_inertia = results[0]["inertia"]

print(
    f"\n{'Algorithm':<14} | {'Runtime (s)':<11} | {'Speedup':<9} | "
    f"{'Inertia':<10} | {'Rel. Inertia'}"
)
print("-" * 65)
for r in results:
    speedup = pam_time / r["time"]
    rel_in = (r["inertia"] / pam_inertia) * 100.0
    print(
        f"{r['name']:<14} | {r['time']:>9.4f} s | {speedup:>7.1f}x | "
        f"{r['inertia']:>10.2f} | {rel_in:>9.1f}%"
    )

# ---------------------------------------------------------------------------
# Multi-panel visualization
# ---------------------------------------------------------------------------
labels = [r["label"] for r in results]
times = [r["time"] for r in results]
inertias = [r["inertia"] for r in results]
colors = [r["color"] for r in results]
rel_inertias = [(r["inertia"] / pam_inertia) * 100.0 for r in results]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.2, 4.2))

# 1. Runtime comparison on Logarithmic scale
bars1 = ax1.bar(labels, times, color=colors, width=0.55, edgecolor="#555555", linewidth=0.8)
ax1.set_yscale("log")
ax1.set_ylabel("Runtime (seconds, log scale)", fontsize=10)
ax1.set_title("Execution Time (Lower is Faster)\nLog Scale ($\\log_{10}$)", fontsize=11, pad=10)
ax1.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#e5e5e5", alpha=0.8)
ax1.set_axisbelow(True)

for bar, t in zip(bars1, times):
    speedup = pam_time / t
    txt = f"{t:.3f}s\n({speedup:.1f}x)" if speedup >= 1.5 else f"{t:.3f}s\n(1.0x)"
    ax1.annotate(
        txt,
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

# 2. Quality comparison (Relative Inertia %)
bars2 = ax2.bar(
    labels,
    rel_inertias,
    color=colors,
    width=0.55,
    edgecolor="#555555",
    linewidth=0.8,
)
ax2.set_ylabel("Inertia / Total Distance (% of PAM, Lower is Better)", fontsize=10)
ax2.set_title("Inertia Loss (Lower is Better)\nRelative to PAM Baseline", fontsize=11, pad=10)
ax2.set_ylim(95, max(rel_inertias) + 6)
ax2.axhline(
    100.0,
    color="#7570b3",
    linestyle="--",
    linewidth=1.2,
    alpha=0.8,
    label="PAM baseline (100% = Best)",
)
ax2.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#e5e5e5", alpha=0.8)
ax2.set_axisbelow(True)

for bar, rel in zip(bars2, rel_inertias):
    diff = rel - 100.0
    if diff <= 0.05:
        txt = f"{rel:.1f}%\n(Best/Opt)"
    else:
        txt = f"{rel:.1f}%\n(+{diff:.1f}% loss)"
    ax2.annotate(
        txt,
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
ax2.legend(loc="upper left", frameon=False, fontsize=8.5)

# 3. 2D cluster partitions and overlaid medoid locations
fast_model = results[3]["model"]
ax3.scatter(
    X[:, 0],
    X[:, 1],
    c=fast_model.labels_,
    cmap="tab10",
    s=8,
    alpha=0.35,
    edgecolors="none",
)

pam_centers = results[0]["model"].cluster_centers_
ax3.scatter(
    pam_centers[:, 0],
    pam_centers[:, 1],
    marker="o",
    s=160,
    facecolors="none",
    edgecolors="#7570b3",
    linewidths=2.2,
    label="PAM Medoids",
    zorder=5,
)

clara_centers = results[1]["model"].cluster_centers_
ax3.scatter(
    clara_centers[:, 0],
    clara_centers[:, 1],
    marker="^",
    s=110,
    c="#e7298a",
    linewidths=1.2,
    edgecolors="black",
    label="CLARA Medoids",
    zorder=6,
)

fast_centers = fast_model.cluster_centers_
ax3.scatter(
    fast_centers[:, 0],
    fast_centers[:, 1],
    marker="x",
    s=110,
    c="black",
    linewidths=2.2,
    label="FastCLARANS",
    zorder=7,
)

ax3.set_title(f"Medoid Placement ($N={n_samples}, K={n_clusters}$)", fontsize=11, pad=10)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

plt.tight_layout()
plt.show()
