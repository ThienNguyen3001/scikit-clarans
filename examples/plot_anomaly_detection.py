"""
==============================================================
Unsupervised Anomaly Detection via Medoid Distance Scoring
==============================================================

This example demonstrates how CLARANS and FastCLARANS can be used for
unsupervised anomaly and outlier detection.

Because k-medoids selects actual exemplar data points as medoids and minimizes
linear distances rather than squared penalties, its cluster centers are resistant
to being dragged away by extreme values. Consequently, the distance from any
sample :math:`x` to its nearest cluster medoid:

.. math::

    \\text{anomaly\\_score}(x) = \\min_{j=1,\\dots,k} d(x, m_j)

serves as a robust, interpretable anomaly score. In scikit-clarans, this can be
computed directly via:

.. code-block:: python

    scores = model.transform(X).min(axis=1)

Samples exceeding a quantile threshold (such as the 95th percentile) are flagged
as potential anomalies or sensor faults.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from clarans import FastCLARANS

COLORS = ["#2b5c8f", "#d95f02", "#7570b3"]


# 1. Generate regular clustered inliers + background uniform anomalies
rng = np.random.RandomState(42)
n_inliers = 400
n_outliers = 25

X_inliers, _ = make_blobs(
    n_samples=n_inliers,
    centers=[(-3, -2), (2, 3), (4, -3)],
    cluster_std=[0.65, 0.60, 0.70],
    random_state=42,
)

# Outliers distributed sparsely across the domain
X_outliers = rng.uniform(low=[-7, -7], high=[8, 8], size=(n_outliers, 2))
X = np.vstack([X_inliers, X_outliers])
y_true = np.array([0] * n_inliers + [1] * n_outliers)

# 2. Fit FastCLARANS on the combined data
model = FastCLARANS(n_clusters=3, num_local=5, random_state=42)
model.fit(X)

# 3. Compute anomaly scores as distance to nearest medoid
# transform(X) returns (n_samples, n_clusters) pairwise distances to each medoid
dist_to_all_medoids = model.transform(X)
anomaly_scores = dist_to_all_medoids.min(axis=1)

# Establish threshold at 95th percentile
threshold = np.percentile(anomaly_scores, 95)
predicted_anomalies = anomaly_scores >= threshold

n_detected = np.sum(predicted_anomalies)
print(f"95th Percentile Anomaly Threshold: {threshold:.2f}")
print(f"Total Detected Anomalies:         {n_detected} / {len(X)}")

# 4. Visualization: (a) Decision Surface & Detections, (b) Score Distribution
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

# Panel (a): Scatter with Anomaly Score Contours
ax_a = axes[0]
grid_x, grid_y = np.meshgrid(
    np.linspace(-8, 9, 120),
    np.linspace(-8, 9, 120),
)
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
grid_scores = model.transform(grid_points).min(axis=1).reshape(grid_x.shape)

contour = ax_a.contourf(grid_x, grid_y, grid_scores, levels=14, cmap="Blues_r", alpha=0.35)
cbar = plt.colorbar(contour, ax=ax_a, fraction=0.046, pad=0.04)
cbar.set_label("Anomaly Score $\\min d(x, m)$", fontsize=8.5)

# Plot inliers
for k in range(3):
    inlier_mask = (model.labels_ == k) & (~predicted_anomalies)
    ax_a.scatter(
        X[inlier_mask, 0],
        X[inlier_mask, 1],
        c=COLORS[k],
        s=22,
        alpha=0.75,
        edgecolors="none",
        label=f"Cluster {k + 1}" if k == 0 else None,
    )

# Plot flagged anomalies
ax_a.scatter(
    X[predicted_anomalies, 0],
    X[predicted_anomalies, 1],
    c="#e74c3c",
    marker="^",
    s=55,
    linewidths=1.2,
    edgecolors="black",
    label=f"Flagged Anomalies ({n_detected})",
    zorder=4,
)

# Plot medoids
medoids = model.cluster_centers_
ax_a.scatter(
    medoids[:, 0],
    medoids[:, 1],
    marker="X",
    s=130,
    c="black",
    linewidths=1.8,
    label="Cluster Medoids",
    zorder=5,
)

ax_a.set_title("(a) Outlier Detection via Medoid Distance", fontsize=10.5, pad=8)
ax_a.set_xticks([])
ax_a.set_yticks([])
for spine in ax_a.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)
ax_a.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.0)

# Panel (b): Score Histogram & Cutoff Threshold
ax_b = axes[1]
n_bins, bins, patches = ax_b.hist(
    anomaly_scores,
    bins=30,
    color="#2b5c8f",
    alpha=0.7,
    edgecolor="white",
    density=False,
)

# Color bars above threshold red
for bin_left, patch in zip(bins[:-1], patches):
    if bin_left >= threshold:
        patch.set_facecolor("#e74c3c")

ax_b.axvline(
    threshold,
    color="#c0392b",
    linestyle="--",
    linewidth=1.8,
    label=f"Cutoff = {threshold:.2f} (95th pct)",
)
ax_b.set_title("(b) Anomaly Score Distribution", fontsize=10.5, pad=8)
ax_b.set_xlabel("Anomaly Score (Distance to Nearest Medoid)", fontsize=9.0)
ax_b.set_ylabel("Sample Count", fontsize=9.0)
for spine in ax_b.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)
ax_b.legend(loc="upper right", frameon=False, fontsize=8.5)

plt.suptitle(
    "Unsupervised Anomaly Scoring: model.transform(X).min(axis=1)",
    fontsize=12,
    y=0.98,
)
plt.tight_layout()
plt.show()
