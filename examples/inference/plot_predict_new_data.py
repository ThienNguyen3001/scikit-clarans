"""
==============================================
Predicting Cluster Labels on Unseen Query Data
==============================================

This example fits CLARANS on training data and visualizes how new, unseen
samples are assigned to their nearest fitted cluster medoids.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clarans import CLARANS

X_train, _ = make_blobs(n_samples=400, centers=3, n_features=2, random_state=42)
model = CLARANS(n_clusters=3, init="k-medoids++", random_state=42)
model.fit(X_train)

# Query points
rng = np.random.RandomState(99)
X_query = rng.uniform(
    low=[X_train[:, 0].min(), X_train[:, 1].min()],
    high=[X_train[:, 0].max(), X_train[:, 1].max()],
    size=(40, 2),
)
query_preds = model.predict(X_query)

fig, ax = plt.subplots(figsize=(8.2, 5))

# Training data
ax.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=model.labels_,
    cmap="tab10",
    s=20,
    alpha=0.45,
    label="Train Data",
)

# Predicted query points
ax.scatter(
    X_query[:, 0],
    X_query[:, 1],
    c=query_preds,
    cmap="tab10",
    marker="s",
    s=70,
    edgecolors="black",
    linewidths=1.2,
    label="Predicted Query",
)

# Medoids
ax.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=160,
    c="black",
    linewidths=2,
    label="Medoids",
    zorder=10,
)

for spine in ax.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

ax.set_xticks([])
ax.set_yticks([])

ax.set_title("CLARANS Cluster Predictions for New Unseen Samples", fontsize=11, pad=8)
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=8.5,
)
plt.tight_layout()
plt.show()
