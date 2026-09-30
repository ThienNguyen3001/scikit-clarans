r"""
===========================================================
Comparing Cost Evaluation Strategies: Delta vs. Brute-Force
===========================================================

This example benchmarks the two swap evaluation strategies in CLARANS:
``cost_evaluation='delta'`` (default) and ``cost_evaluation='brute_force'``.

Ng & Han (2002) originally evaluated each candidate swap by recalculating the
full clustering cost in :math:`O(n \cdot k \cdot d)`. In contrast, delta caching
maintains the nearest (:math:`d_1`) and second-nearest (:math:`d_2`) medoid
distances for all samples, evaluating candidate swaps in :math:`O(n \cdot d)`.

Both strategies produce identical clustering decisions and final inertia,
while delta caching provides a significant speedup that grows with :math:`k`.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS

# Generate synthetic dataset
n_samples = 1000
k_values = [4, 8, 12]
X, _ = make_blobs(n_samples=n_samples, centers=12, n_features=2, random_state=42)

times_delta = []
times_brute = []
inertias_delta = []
inertias_brute = []

for k in k_values:
    # Benchmark delta caching
    t0 = time.perf_counter()
    model_delta = CLARANS(
        n_clusters=k,
        num_local=2,
        cost_evaluation="delta",
        max_neighbors=150,
        random_state=42,
    )
    model_delta.fit(X)
    times_delta.append(time.perf_counter() - t0)
    inertias_delta.append(model_delta.inertia_)

    # Benchmark brute-force recalculation
    t0 = time.perf_counter()
    model_brute = CLARANS(
        n_clusters=k,
        num_local=2,
        cost_evaluation="brute_force",
        max_neighbors=150,
        random_state=42,
    )
    model_brute.fit(X)
    times_brute.append(time.perf_counter() - t0)
    inertias_brute.append(model_brute.inertia_)

# Compute speedup factors
speedups = [b / d for b, d in zip(times_brute, times_delta)]

# Verify solution equivalence
all_equal = all(np.isclose(d, b, rtol=1e-5) for d, b in zip(inertias_delta, inertias_brute))
print(f"Solutions identical across all k: {all_equal}")
for k, d_cost, b_cost in zip(k_values, inertias_delta, inertias_brute):
    print(f"  k={k:2d}: delta={d_cost:.4f}, brute_force={b_cost:.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

x = np.arange(len(k_values))
width = 0.35

# Panel 1: Execution runtime
bars1 = ax1.bar(x - width / 2, times_delta, width, label="Delta caching (O(n))", color="#4c72b0")
bars2 = ax1.bar(x + width / 2, times_brute, width, label="Brute-force (O(n*k))", color="#c44e52")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Runtime (seconds)")
ax1.set_title("Runtime Comparison (Lower is Faster)")
ax1.set_xticks(x)
ax1.set_xticklabels([f"k={k}" for k in k_values])
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.2f}s", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.2f}s", ha="center", va="bottom", fontsize=9)

# Panel 2: Speedup factor
bars_speedup = ax2.bar(x, speedups, width=0.45, color="#55a868")
ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax2.set_xlabel("Number of clusters (k)")
ax2.set_ylabel("Speedup factor (Brute / Delta)")
ax2.set_title("Delta Caching Speedup")
ax2.set_xticks(x)
ax2.set_xticklabels([f"k={k}" for k in k_values])
ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

for bar in bars_speedup:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, y + 0.05, f"{y:.2f}x", ha="center", va="bottom", fontweight="bold")

plt.suptitle(f"CLARANS Cost Evaluation Strategies on N={n_samples}", fontsize=13)
plt.tight_layout()
plt.show()
