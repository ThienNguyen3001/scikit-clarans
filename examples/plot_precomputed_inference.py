"""
========================================================================
Fast Precomputed Inference: Compact (n_test, k) vs. Full (n_test, n_train)
========================================================================

This example demonstrates an optimization in scikit-clarans for
inference (``predict`` and ``transform``) on precomputed distance matrices.

When serving models in production with precomputed distances (such as graph
shortest paths, custom kernels, or expensive non-Euclidean distance functions),
the naive approach requires computing distances between every new query sample
and **all** :math:`n_{\\text{train}}` training samples (matrix shape:
:math:`(n_{\\text{test}}, n_{\\text{train}})`).

However, assigning a new query sample to its nearest cluster depends **only**
on its distances to the :math:`k` medoids:

.. math::

    \\text{label}(x) = \\arg\\min_{j=1,\\dots,k} d(x, m_j)

scikit-clarans accepts either matrix shape:

1. **Full matrix**: :math:`(n_{\\text{test}}, n_{\\text{train}})`
2. **Compact matrix**: :math:`(n_{\\text{test}}, k)` matching the fitted medoids directly.

Using the compact matrix reduces distance evaluations and memory consumption
by a factor of :math:`n_{\\text{train}} / k` (typically 10x to 1000x faster)
while producing identical predictions.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from clarans import FastCLARANS

# 1. Generate training and query test data
rng = np.random.RandomState(42)
n_train = 800
n_test = 300
n_clusters = 4

X_train, _ = make_blobs(n_samples=n_train, centers=n_clusters, n_features=6, random_state=42)
X_test, _ = make_blobs(n_samples=n_test, centers=n_clusters, n_features=6, random_state=123)

# 2. Fit FastCLARANS on training pairwise distance matrix
D_train = pairwise_distances(X_train, metric="euclidean")
model = FastCLARANS(n_clusters=n_clusters, metric="precomputed", random_state=42)
model.fit(D_train)

medoid_coords = X_train[model.medoid_indices_]
print(f"Fitted Medoid Indices: {model.medoid_indices_}")

# 3. Strategy A: Full matrix (n_test, n_train)
t0 = time.perf_counter()
D_test_full = pairwise_distances(X_test, X_train, metric="euclidean")
preds_full = model.predict(D_test_full)
time_full = time.perf_counter() - t0
elements_full = D_test_full.size

# 4. Strategy B: Compact matrix (n_test, k)
t0 = time.perf_counter()
D_test_compact = pairwise_distances(X_test, medoid_coords, metric="euclidean")
preds_compact = model.predict(D_test_compact)
time_compact = time.perf_counter() - t0
elements_compact = D_test_compact.size

# Verify exact equivalence
assert np.array_equal(preds_full, preds_compact)

speedup = time_full / time_compact
mem_ratio = elements_full / elements_compact
print(f"Full Matrix Evaluations:       {elements_full:,} entries ({time_full * 1e3:.2f} ms)")
print(f"Compact Matrix Evaluations:    {elements_compact:,} entries ({time_compact * 1e3:.2f} ms)")
print(f"Distance Computation Speedup:  {speedup:.1f}x faster")
print(f"Memory Reduction:              {mem_ratio:.0f}x smaller")
print(f"Prediction Equivalence:        {np.array_equal(preds_full, preds_compact)}")

# 5. Visualization: Benchmark Comparison & Cluster Assignment
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# Panel (a): Distance Evaluations & Memory Comparison
ax_a = axes[0]
strategies = ["Full Matrix\n$(n_{test}, n_{train})$", "Compact Matrix\n$(n_{test}, k)$"]
eval_counts = [elements_full, elements_compact]
colors = ["#7f8c8d", "#2b5c8f"]

bars = ax_a.bar(strategies, eval_counts, color=colors, width=0.45)
ax_a.set_yscale("log")
ax_a.set_ylabel("Pairwise Distances Computed (Log Scale)", fontsize=9.0)
ax_a.set_title(
    f"(a) Computational Workload\n{mem_ratio:.0f}x Fewer Distances Computed",
    fontsize=10.5,
    pad=8,
)

for bar in bars:
    height = bar.get_height()
    ax_a.annotate(
        f"{height:,} entries",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
    )

for spine in ax_a.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

# Panel (b): Query Sample Distances to Assigned Medoid
ax_b = axes[1]
# transform(D_test_compact) returns (n_test, k) distance array directly
test_distances = model.transform(D_test_compact)
min_dists = test_distances.min(axis=1)

palette = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
for k in range(n_clusters):
    mask = preds_compact == k
    ax_b.scatter(
        np.where(mask)[0],
        min_dists[mask],
        c=palette[k],
        s=25,
        alpha=0.75,
        label=f"Cluster {k + 1} (Medoid {model.medoid_indices_[k]})",
    )

ax_b.set_title(
    f"(b) Assigned Cluster Distance for {n_test} Query Samples\n(100% Identical Predictions)",
    fontsize=10.5,
    pad=8,
)
ax_b.set_xlabel("Query Sample Index", fontsize=9.0)
ax_b.set_ylabel("Distance to Nearest Medoid", fontsize=9.0)
for spine in ax_b.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)
ax_b.legend(loc="upper right", frameon=False, fontsize=8.0)

plt.suptitle(
    "Compact (n_test, k) vs. Full (n_test, n_train) Precomputed Inference",
    fontsize=12,
    y=0.98,
)
plt.tight_layout()
plt.show()
