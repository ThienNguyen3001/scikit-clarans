"""
======================================================
Benchmarking CLARANS vs. FastCLARANS Convergence Speed
======================================================

This example compares the execution runtime and resulting inertia between
classic CLARANS (Ng & Han, 2002) and FastCLARANS (Schubert & Rousseeuw, 2021).
FastCLARANS uses vectorized FastPAM1 delta evaluations to test all k medoid
swaps simultaneously.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS, FastCLARANS

# 1. Generate benchmark dataset
n_samples = 1500
n_clusters = 6
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.8, random_state=42)

# 2. Benchmark CLARANS
t0 = time.perf_counter()
model_clarans = CLARANS(n_clusters=n_clusters, num_local=2, random_state=42)
model_clarans.fit(X)
time_clarans = time.perf_counter() - t0

# 3. Benchmark FastCLARANS
t0 = time.perf_counter()
model_fast = FastCLARANS(n_clusters=n_clusters, num_local=2, random_state=42)
model_fast.fit(X)
time_fast = time.perf_counter() - t0

# 4. Plot comparative summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

algorithms = ["CLARANS", "FastCLARANS"]
runtimes = [time_clarans, time_fast]
inertias = [model_clarans.inertia_, model_fast.inertia_]
colors = ["#4c72b0", "#55a868"]

# Runtime Comparison
bars1 = ax1.bar(algorithms, runtimes, color=colors, width=0.5)
ax1.set_ylabel("Runtime (seconds)")
ax1.set_title("Runtime Comparison (Lower is Faster)")
for bar in bars1:
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.01 * max(runtimes),
        f"{yval:.3f}s",
        ha="center",
        va="bottom",
    )

# Inertia Comparison
bars2 = ax2.bar(algorithms, inertias, color=colors, width=0.5)
ax2.set_ylabel("Inertia (Total Cost)")
ax2.set_title("Inertia Quality Comparison (Lower is Better)")
for bar in bars2:
    yval = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.01 * max(inertias),
        f"{yval:.1f}",
        ha="center",
        va="bottom",
    )

plt.suptitle(f"CLARANS vs. FastCLARANS on N={n_samples}, K={n_clusters}", fontsize=13)
plt.tight_layout()
plt.show()
