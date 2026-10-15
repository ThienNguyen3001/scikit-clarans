"""
=================================================
Feature Transformation via Cluster-Distance Space
=================================================

This example demonstrates using CLARANS as a feature transformer (`transform`),
mapping sample coordinates into distance-to-medoid representations.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS

X, _ = make_blobs(n_samples=250, centers=3, n_features=4, random_state=42)

model = CLARANS(n_clusters=3, random_state=42)
model.fit(X)

# Distances from each sample to fitted medoids
X_trans = model.transform(X)
print(f"Original shape:    {X.shape}")
print(f"Transformed shape: {X_trans.shape}")

# Distances to the first two medoids
fig, ax = plt.subplots(figsize=(6.5, 4.8))
colors = ["#2b5c8f", "#d95f02", "#7570b3"]
for k in range(3):
    mask = model.labels_ == k
    ax.scatter(
        X_trans[mask, 0],
        X_trans[mask, 1],
        c=colors[k],
        s=25,
        alpha=0.65,
        edgecolors="none",
        label=f"Cluster {k + 1}",
    )

for spine in ax.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

ax.set_xlabel("Distance to Medoid 0", fontsize=9.5)
ax.set_ylabel("Distance to Medoid 1", fontsize=9.5)
ax.set_title("Transformed Feature Space: Distance to Medoids", fontsize=11, pad=8)
ax.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=8.5,
)
plt.tight_layout()
plt.show()
