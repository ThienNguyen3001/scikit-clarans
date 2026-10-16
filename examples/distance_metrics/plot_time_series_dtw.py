"""
======================================================
Clustering Time Series with Dynamic Time Warping (DTW)
======================================================

This example shows how FastCLARANS clusters time series using Dynamic
Time Warping (DTW) as a precomputed distance matrix.

Standard Euclidean distance and K-Means struggle on temporal signals for
two main reasons:

1. **Phase shifts and time dilation**: If two waveforms share the same
   underlying shape but are offset in time, Euclidean distance compares
   points at identical time indices, reporting an artificially large error.
2. **Destructive interference in centroids**: K-Means averages curves
   pointwise across the cluster. When curves are out of phase, averaging
   cancels peaks and troughs, producing a flattened prototype that does
   not look like any real signal in the data.

Dynamic Time Warping handles phase shifts by finding an optimal non-linear
alignment between two sequences. FastCLARANS then selects an actual observed
time series (the medoid) as the cluster prototype. The medoid preserves the
true peak height, period, and physical characteristics of the signal.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import inspect
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from clarans import FastCLARANS

COLORS = ["#2b5c8f", "#d95f02", "#1b9e77"]


def dtw_distance_matrix(series: np.ndarray, window: int = 8) -> np.ndarray:
    """Compute pairwise Dynamic Time Warping distance with a Sakoe-Chiba window."""
    n_samples, length = series.shape
    d_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)

    for i in range(n_samples):
        s1 = series[i]
        for j in range(i + 1, n_samples):
            s2 = series[j]
            dp = np.full((length + 1, length + 1), np.inf, dtype=np.float64)
            dp[0, 0] = 0.0

            for r in range(1, length + 1):
                start_c = max(1, r - window)
                end_c = min(length, r + window)
                for c in range(start_c, end_c + 1):
                    cost = abs(s1[r - 1] - s2[c - 1])
                    dp[r, c] = cost + min(dp[r - 1, c], dp[r, c - 1], dp[r - 1, c - 1])

            dist = dp[length, length]
            d_matrix[i, j] = dist
            d_matrix[j, i] = dist

    return d_matrix


# 1. Synthesize 36 time series across 3 distinct waveform families
np.random.seed(42)
n_per_family = 12
length = 40
t = np.linspace(0, 1, length)

curves = []
family_names = ["Gaussian Surge", "Damped Oscillation", "Step Transient"]

# Family 0: Gaussian surge with varying peak center and width
for _ in range(n_per_family):
    mu = np.random.uniform(0.35, 0.60)
    sigma = np.random.uniform(0.08, 0.12)
    noise = np.random.normal(0, 0.03, length)
    curves.append(np.exp(-((t - mu) ** 2) / (2 * sigma**2)) + noise)

# Family 1: Damped oscillation with varying phase and decay
for _ in range(n_per_family):
    phase = np.random.uniform(0.0, 0.8)
    decay = np.random.uniform(2.5, 4.0)
    noise = np.random.normal(0, 0.03, length)
    curves.append(np.exp(-decay * t) * np.sin(4 * np.pi * t + phase) + noise)

# Family 2: Step transient with varying onset time
for _ in range(n_per_family):
    onset = np.random.uniform(0.35, 0.60)
    steepness = np.random.uniform(20.0, 30.0)
    noise = np.random.normal(0, 0.03, length)
    curves.append(1.0 / (1.0 + np.exp(-steepness * (t - onset))) + noise)

curves_arr = np.array(curves)

# 2. Build pairwise DTW dissimilarity matrix
D_dtw = dtw_distance_matrix(curves_arr, window=8)

# 3. Fit FastCLARANS on the precomputed distance matrix
model = FastCLARANS(n_clusters=3, num_local=5, metric="precomputed", random_state=42)
model.fit(D_dtw)

medoids = model.medoid_indices_
print(f"Optimal Medoid Indices: {medoids}")

# 4. Project pairwise DTW distances into 2D using Multidimensional Scaling (MDS)
mds_kwargs = {"n_components": 2, "random_state": 42, "normalized_stress": "auto"}
if "metric_mds" in inspect.signature(MDS).parameters:
    mds_kwargs["metric"] = "precomputed"
    mds_kwargs["init"] = "random"
else:
    mds_kwargs["dissimilarity"] = "precomputed"
mds = MDS(**mds_kwargs)
coords_2d = mds.fit_transform(D_dtw)

# 5. Visualization
fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

# Panel (a): All clusters with FastCLARANS medoids
ax_a = axes[0]
for k in range(3):
    mask = model.labels_ == k
    cluster_curves = curves_arr[mask]
    for curve in cluster_curves:
        ax_a.plot(t, curve, color=COLORS[k], alpha=0.3, linewidth=1.0)

    medoid_curve = curves_arr[model.medoid_indices_[k]]
    ax_a.plot(
        t,
        medoid_curve,
        color=COLORS[k],
        linewidth=2.4,
        label=f"Medoid {k + 1}",
    )

ax_a.set_title("FastCLARANS Clusters + Exemplar Medoids", fontsize=11, pad=8)
ax_a.set_xlabel("Time Index", fontsize=10)
ax_a.set_ylabel("Signal Amplitude", fontsize=10)
ax_a.set_ylim(-0.85, 1.50)
ax_a.legend(
    loc="upper left",
    frameon=True,
    facecolor="white",
    framealpha=0.92,
    edgecolor="#e0e0e0",
    fontsize=8.5,
)
for spine in ax_a.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

# Panel (b): The destructive interference problem of pointwise averaging vs. Medoid
ax_b = axes[1]
# Pick the Gaussian surge cluster to contrast the arithmetic mean vs medoid
gaussian_cluster_idx = int(model.labels_[0])
gaussian_mask = model.labels_ == gaussian_cluster_idx
gaussian_curves = curves_arr[gaussian_mask]
arithmetic_mean = np.mean(gaussian_curves, axis=0)
best_medoid_curve = curves_arr[model.medoid_indices_[gaussian_cluster_idx]]

for curve in gaussian_curves:
    ax_b.plot(t, curve, color="#999999", alpha=0.35, linewidth=1.0)

ax_b.plot(
    t,
    arithmetic_mean,
    color="#d95f02",
    linestyle="--",
    linewidth=2.2,
    label="Pointwise Mean (Centroid)",
)
ax_b.plot(
    t,
    best_medoid_curve,
    color="#2b5c8f",
    linewidth=2.4,
    label="FastCLARANS Medoid",
)

ax_b.set_title("Medoid vs. Pointwise Mean (Averaging Artifact)", fontsize=11, pad=8)
ax_b.set_xlabel("Time Index", fontsize=10)
ax_b.set_yticks([])
ax_b.set_ylim(-0.15, 1.45)
ax_b.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    framealpha=0.92,
    edgecolor="#e0e0e0",
    fontsize=8.5,
)
for spine in ax_b.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

# Panel (c): 2D MDS semantic space of DTW distances
ax_c = axes[2]
for k in range(3):
    mask = model.labels_ == k
    ax_c.scatter(
        coords_2d[mask, 0],
        coords_2d[mask, 1],
        c=COLORS[k],
        s=45,
        alpha=0.7,
        edgecolors="none",
        label=f"Cluster {k + 1}",
    )

med_coords = coords_2d[model.medoid_indices_]
ax_c.scatter(
    med_coords[:, 0],
    med_coords[:, 1],
    marker="*",
    s=220,
    c="gold",
    edgecolors="black",
    linewidths=1.2,
    label="Medoids",
    zorder=5,
)

ax_c.set_title("2D Projection of DTW Distance Space", fontsize=11, pad=8)
ax_c.set_xticks([])
ax_c.set_yticks([])
ax_c.set_ylim(-6.8, 5.8)
ax_c.legend(
    loc="upper center",
    ncol=2,
    frameon=True,
    facecolor="white",
    framealpha=0.92,
    edgecolor="#e0e0e0",
    fontsize=8.5,
)
for spine in ax_c.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.show()
