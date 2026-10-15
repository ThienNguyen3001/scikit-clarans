"""
====================================================
Clustering Incomplete Data with Missing Values (NaN)
====================================================

This example demonstrates how CLARANS and FastCLARANS cluster feature matrices
containing missing values (NaN) using ``metric='nan_euclidean'``.

.. note::

    **How ``nan_euclidean`` handles missing attributes:**
    When computing pairwise distances between two samples :math:`u` and :math:`v`,
    any dimension where either vector contains a NaN is omitted from the Euclidean
    sum. The accumulated squared distance across non-missing dimensions is then
    scaled up proportionally:

    .. math::

        d(u, v) = \\sqrt{\\frac{d}{d_{\\text{valid}}} \\sum_{i \\in \\text{valid}} (u_i - v_i)^2}

    where :math:`d` is the total number of features and :math:`d_{\\text{valid}}`
    is the number of mutually observed dimensions. This prevents sample-pair
    distance distortions caused by differing counts of missing coordinates.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from clarans import CLARANS, FastCLARANS

COLORS = ["#2b5c8f", "#d95f02", "#7570b3"]


# 1. Generate synthetic Gaussian blobs
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.65, random_state=42)

# 2. Introduce ~15% missing values (NaN) into the dataset
rng = np.random.RandomState(42)
nan_mask = rng.rand(*X.shape) < 0.15
# Ensure each sample retains at least one observed feature
nan_mask[nan_mask.all(axis=1), 1] = False

X_nan = X.copy()
X_nan[nan_mask] = np.nan

# 3. Fit baseline CLARANS on complete data
model_complete = CLARANS(n_clusters=3, num_local=5, metric="euclidean", random_state=42)
model_complete.fit(X)

# 4. Fit FastCLARANS on incomplete data with metric='nan_euclidean'
model_nan = FastCLARANS(n_clusters=3, num_local=5, metric="nan_euclidean", random_state=42)
model_nan.fit(X_nan)

# Evaluate partition similarity between complete and missing data fits
ari = adjusted_rand_score(model_complete.labels_, model_nan.labels_)
print(f"Complete Data Inertia:         {model_complete.inertia_:.2f}")
print(f"Missing Data (NaN) Inertia:    {model_nan.inertia_:.2f}")
print(f"Adjusted Rand Index (ARI):     {ari:.4f}")

# 5. Visualization
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

# Subplot (a): Complete Data Baseline
ax_a = axes[0]
for k in range(3):
    mask = model_complete.labels_ == k
    ax_a.scatter(
        X[mask, 0],
        X[mask, 1],
        c=COLORS[k],
        s=25,
        alpha=0.65,
        edgecolors="none",
    )

centers_a = model_complete.cluster_centers_
ax_a.scatter(
    centers_a[:, 0],
    centers_a[:, 1],
    marker="X",
    s=110,
    c="black",
    linewidths=1.5,
    label="Medoids",
    zorder=4,
)
ax_a.set_title(
    f"(a) Complete Data Baseline\nInertia = {model_complete.inertia_:.1f}",
    fontsize=10.5,
    pad=8,
)
ax_a.set_xticks([])
ax_a.set_yticks([])
for spine in ax_a.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)
ax_a.legend(loc="upper right", frameon=False, fontsize=9)

# Subplot (b): Missing Data with nan_euclidean
ax_b = axes[1]
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
ax_b.set_xlim(x_min, x_max)
ax_b.set_ylim(y_min, y_max)

valid_2d = ~np.isnan(X_nan).any(axis=1)
missing_y = ~np.isnan(X_nan[:, 0]) & np.isnan(X_nan[:, 1])
missing_x = np.isnan(X_nan[:, 0]) & ~np.isnan(X_nan[:, 1])

for k in range(3):
    # Fully observed 2D samples
    mask_valid = (model_nan.labels_ == k) & valid_2d
    ax_b.scatter(
        X_nan[mask_valid, 0],
        X_nan[mask_valid, 1],
        c=COLORS[k],
        s=25,
        alpha=0.65,
        edgecolors="none",
    )

    # Samples with missing y-coordinate plotted as marginal ticks at the bottom
    mask_miss_y = (model_nan.labels_ == k) & missing_y
    if np.any(mask_miss_y):
        ax_b.scatter(
            X_nan[mask_miss_y, 0],
            np.full(np.sum(mask_miss_y), y_min + 0.2),
            c=COLORS[k],
            marker="|",
            s=80,
            linewidths=1.8,
            label="Missing $y$" if k == 0 else None,
            zorder=3,
        )

    # Samples with missing x-coordinate plotted as marginal ticks on the left
    mask_miss_x = (model_nan.labels_ == k) & missing_x
    if np.any(mask_miss_x):
        ax_b.scatter(
            np.full(np.sum(mask_miss_x), x_min + 0.2),
            X_nan[mask_miss_x, 1],
            c=COLORS[k],
            marker="_",
            s=80,
            linewidths=1.8,
            label="Missing $x$" if k == 0 else None,
            zorder=3,
        )

centers_b = model_nan.cluster_centers_
ax_b.scatter(
    centers_b[:, 0],
    centers_b[:, 1],
    marker="X",
    s=110,
    c="black",
    linewidths=1.5,
    label="Medoids",
    zorder=4,
)

ax_b.set_title(
    f"(b) Incomplete Data (15% NaN, metric='nan_euclidean')\nARI vs Baseline = {ari:.2f}",
    fontsize=10.5,
    pad=8,
)
ax_b.set_xticks([])
ax_b.set_yticks([])
for spine in ax_b.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)
ax_b.legend(loc="upper right", frameon=False, fontsize=8.5)

plt.suptitle(
    "Native Handling of Missing Values (NaN) via metric='nan_euclidean'",
    fontsize=12,
    y=0.98,
)
plt.tight_layout()
plt.show()
