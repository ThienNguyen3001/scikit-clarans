"""
==============================================================
Clustering Non-Vector Data: Text Sequences with Edit Distance
==============================================================

This example demonstrates how CLARANS and FastCLARANS cluster non-vector
objects (such as text strings, search queries, DNA/amino-acid motifs) where
traditional centroid-based algorithms like K-Means fail.

In non-Euclidean spaces, computing a "mean" or "centroid" is ill-defined:
what is the average string between ``"biology"`` and ``"computer"``?

K-Medoids solves this by identifying actual exemplar objects (**Medoid Strings**)
from the input collection that minimize the total edit distance to all other
words in their cluster.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from clarans import FastCLARANS

COLORS = ["#2b5c8f", "#d95f02", "#1b9e77"]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings using dynamic programming."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# 1. Non-vector dataset: 18 vocabulary words across 3 morphological families
words = [
    # Biological family
    "biology",
    "biological",
    "biologist",
    "bioinformatics",
    "biomedical",
    "biotechnology",
    # Computing family
    "computer",
    "computing",
    "computation",
    "computational",
    "compiler",
    "compute",
    # Clustering family
    "cluster",
    "clustering",
    "clusters",
    "clustered",
    "classifier",
    "classify",
]

n = len(words)
# 2. Build pairwise Levenshtein distance matrix
D = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    for j in range(n):
        D[i, j] = levenshtein_distance(words[i], words[j])

# 3. Fit FastCLARANS on the precomputed distance matrix
model = FastCLARANS(n_clusters=3, num_local=5, metric="precomputed", random_state=42)
model.fit(D)

medoid_words = [words[idx] for idx in model.medoid_indices_]
print(f"Optimal Medoid Words: {medoid_words}")
for k, med_idx in enumerate(model.medoid_indices_):
    members = [words[i] for i in range(n) if model.labels_[i] == k]
    print(f"Cluster {k + 1} (Medoid: '{words[med_idx]}'): {members}")

# 4. Project pairwise distances into 2D via Multidimensional Scaling (MDS)
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    init="random",
    random_state=42,
    normalized_stress="auto",
)
coords_2d = mds.fit_transform(D)

# 5. Visualization
fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

# Panel (a): 2D MDS semantic space with text annotations
ax_a = axes[0]
for k in range(3):
    mask = model.labels_ == k
    ax_a.scatter(
        coords_2d[mask, 0],
        coords_2d[mask, 1],
        c=COLORS[k],
        s=70,
        alpha=0.65,
        edgecolors="none",
        label=f"Cluster {k + 1} ('{words[model.medoid_indices_[k]]}')",
    )

# Highlight medoid words with gold stars
med_coords = coords_2d[model.medoid_indices_]
ax_a.scatter(
    med_coords[:, 0],
    med_coords[:, 1],
    marker="*",
    s=240,
    c="gold",
    edgecolors="black",
    linewidths=1.5,
    label="Medoid Words",
    zorder=5,
)

# Annotate each word in the 2D projection
for i, word in enumerate(words):
    is_medoid = i in model.medoid_indices_
    weight = "bold" if is_medoid else "normal"
    ax_a.annotate(
        word,
        (coords_2d[i, 0], coords_2d[i, 1]),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=8.0,
        fontweight=weight,
    )

ax_a.set_title("(a) 2D MDS Projection of Levenshtein Distances", fontsize=10.5, pad=8)
ax_a.set_xticks([])
ax_a.set_yticks([])
for spine in ax_a.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)
ax_a.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.0)

# Panel (b): Permuted Distance Matrix Ordered by Cluster
sort_idx = np.argsort(model.labels_)
D_sorted = D[sort_idx, :][:, sort_idx]
sorted_words = [words[i] for i in sort_idx]

ax_b = axes[1]
im_b = ax_b.imshow(D_sorted, cmap="Blues_r", origin="upper")
ax_b.set_xticks(range(n))
ax_b.set_yticks(range(n))
ax_b.set_xticklabels(sorted_words, rotation=60, ha="right", fontsize=7.5)
ax_b.set_yticklabels(sorted_words, fontsize=7.5)
ax_b.set_title("(b) Clustered Levenshtein Matrix", fontsize=10.5, pad=8)
plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04, label="Edit Distance")

plt.suptitle(
    "Clustering Non-Vector Strings: Representative Medoid Words",
    fontsize=12,
    y=0.98,
)
plt.tight_layout()
plt.show()
