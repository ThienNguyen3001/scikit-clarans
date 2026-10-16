"""
===================================================
User-Provided Initial Candidate Centers for CLARANS
===================================================

This example demonstrates how to pass explicit domain-specific coordinates
as the initial medoids (`init=array_like`), which CLARANS automatically snaps
to the nearest actual samples in the training dataset.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# Initial seed coordinates
user_seeds = np.array([[-5.0, -5.0], [0.0, 5.0], [5.0, -2.0]])

model = CLARANS(n_clusters=3, init=user_seeds, num_local=1, random_state=42)
model.fit(X)

fig, ax = plt.subplots(figsize=(6.5, 4.8))
colors = ["#2b5c8f", "#d95f02", "#7570b3"]
for k in range(3):
    mask = model.labels_ == k
    ax.scatter(
        X[mask, 0],
        X[mask, 1],
        c=colors[k],
        s=25,
        alpha=0.6,
        edgecolors="none",
        label=f"Cluster {k + 1}",
    )

# Seed coordinates
ax.scatter(
    user_seeds[:, 0],
    user_seeds[:, 1],
    c="#e74c3c",
    marker="^",
    s=130,
    linewidths=1.5,
    edgecolors="black",
    label="User Input Seeds",
    zorder=4,
)

# Final medoids
ax.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=150,
    linewidths=2,
    label="Fitted Medoids",
    zorder=10,
)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

ax.set_title("CLARANS with User-Provided Initial Seeds", fontsize=11, pad=8)
ax.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=8.5,
)
plt.tight_layout()
plt.show()
