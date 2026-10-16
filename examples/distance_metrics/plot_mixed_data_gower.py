"""
=============================================================
Clustering Mixed-Type Tabular Data with Gower's Dissimilarity
=============================================================

This example shows how FastCLARANS clusters mixed-type tabular data
combining continuous, binary, and categorical attributes using Gower's
dissimilarity.

Real-world datasets—such as customer records, patient diagnostics, and
insurance accounts—regularly mix different attribute types:

1. **Continuous variables**: Age, annual income, account tenure.
2. **Binary indicators**: Active credit card holder (0 or 1).
3. **Categorical / Nominal features**: Preferred banking channel
   (Branch, Online, Mobile).

Standard K-Means with one-hot encoding struggles on mixed tables:

- **Distance distortion**: One-hot categorical indicators distort Euclidean
  distances, allowing continuous variables with wider numeric ranges to
  dominate or misalign cluster boundaries.
- **Fractional phantom centroids**: Taking the arithmetic mean across mixed
  columns creates artificial centroids with nonsensical values (for example,
  ``Channel: Mobile = 0.42``, ``Credit Card = 0.65``). No real customer has
  42% of a mobile channel.

Gower's dissimilarity (Gower, 1971) evaluates each feature type on its own
terms: continuous variables are scaled by their range, while categorical and
binary variables are scored by exact matching. FastCLARANS then identifies
actual customer records (**Medoid Personas**). Every medoid is a real
individual with valid discrete choices and exact continuous numbers.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import inspect
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from clarans import FastCLARANS

COLORS = ["#2b5c8f", "#d95f02", "#1b9e77"]


def gower_distance_matrix(X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
    """Compute pairwise Gower dissimilarity across continuous and categorical features.

    Continuous features use normalized Manhattan distance. Categorical and binary
    features use simple matching (0 for identical categories, 1 for different).
    """
    ranges = np.ptp(X_num, axis=0)
    ranges[ranges == 0] = 1.0

    # Normalized absolute differences for continuous variables
    diff_num = np.abs(X_num[:, None, :] - X_num[None, :, :]) / ranges
    dist_num = np.sum(diff_num, axis=-1)

    # Simple matching distance for discrete variables
    diff_cat = (X_cat[:, None, :] != X_cat[None, :, :]).astype(float)
    dist_cat = np.sum(diff_cat, axis=-1)

    total_features = X_num.shape[1] + X_cat.shape[1]
    return (dist_num + dist_cat) / total_features


# 1. Synthesize 75 customer records across 3 distinct consumer segments
np.random.seed(42)
n_per_segment = 25

# Segment 0: Young Tech Professionals (higher income, mobile users)
age_0 = np.random.normal(29, 3, n_per_segment).clip(22, 38)
income_0 = np.random.normal(120, 12, n_per_segment).clip(90, 160)
tenure_0 = np.random.normal(18, 5, n_per_segment).clip(6, 32)
# Channel: 0=Branch, 1=Online, 2=Mobile
channel_0 = np.random.choice([0, 1, 2], p=[0.05, 0.15, 0.80], size=n_per_segment)
card_0 = np.random.choice([0, 1], p=[0.05, 0.95], size=n_per_segment)

# Segment 1: Mature Traditionalists (retirees, branch visitors, long tenure)
age_1 = np.random.normal(64, 4, n_per_segment).clip(52, 75)
income_1 = np.random.normal(68, 10, n_per_segment).clip(45, 90)
tenure_1 = np.random.normal(88, 12, n_per_segment).clip(60, 120)
channel_1 = np.random.choice([0, 1, 2], p=[0.80, 0.15, 0.05], size=n_per_segment)
card_1 = np.random.choice([0, 1], p=[0.75, 0.25], size=n_per_segment)

# Segment 2: Budget Students (young age, lower income, online banking)
age_2 = np.random.normal(21, 2, n_per_segment).clip(18, 25)
income_2 = np.random.normal(32, 6, n_per_segment).clip(18, 45)
tenure_2 = np.random.normal(9, 3, n_per_segment).clip(1, 18)
channel_2 = np.random.choice([0, 1, 2], p=[0.10, 0.80, 0.10], size=n_per_segment)
card_2 = np.random.choice([0, 1], p=[0.60, 0.40], size=n_per_segment)

ages = np.concatenate([age_0, age_1, age_2])
incomes = np.concatenate([income_0, income_1, income_2])
tenures = np.concatenate([tenure_0, tenure_1, tenure_2])
channels = np.concatenate([channel_0, channel_1, channel_2])
cards = np.concatenate([card_0, card_1, card_2])

X_num = np.column_stack([ages, incomes, tenures])
X_cat = np.column_stack([channels, cards])

channel_names = ["Branch", "Online", "Mobile"]
card_names = ["No", "Yes"]

# 2. Build pairwise Gower dissimilarity matrix
D_gower = gower_distance_matrix(X_num, X_cat)

# 3. Fit FastCLARANS on the precomputed distance matrix
model = FastCLARANS(n_clusters=3, num_local=5, metric="precomputed", random_state=42)
model.fit(D_gower)

print(f"Optimal Medoid Customer Indices: {model.medoid_indices_}")
for k, idx in enumerate(model.medoid_indices_):
    print(
        f"Medoid Persona {k + 1} (Customer #{idx}): "
        f"Age={ages[idx]:.0f}, Income=${incomes[idx]:.0f}k, "
        f"Tenure={tenures[idx]:.0f}m, Channel='{channel_names[channels[idx]]}', "
        f"CreditCard='{card_names[cards[idx]]}'"
    )

# 4. Project Gower distance matrix into 2D via Multidimensional Scaling (MDS)
mds_kwargs = {"n_components": 2, "random_state": 42, "normalized_stress": "auto"}
if "metric_mds" in inspect.signature(MDS).parameters:
    mds_kwargs["metric"] = "precomputed"
    mds_kwargs["init"] = "random"
else:
    mds_kwargs["dissimilarity"] = "precomputed"
mds = MDS(**mds_kwargs)
coords_2d = mds.fit_transform(D_gower)

# 5. Visualization
fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))

# Panel (a): Parallel attribute profile of the 3 Medoid Personas
ax_a = axes[0]
feature_labels = ["Age", "Income", "Tenure", "Channel", "Credit Card"]
# Normalize all 5 features to [0, 1] for visual comparison
ranges_num = np.ptp(X_num, axis=0)
norm_num = (X_num - X_num.min(axis=0)) / ranges_num
norm_channel = channels / 2.0
norm_card = cards.astype(float)
norm_all = np.column_stack([norm_num, norm_channel, norm_card])

x_coords = np.arange(len(feature_labels))
for k, idx in enumerate(model.medoid_indices_):
    medoid_profile = norm_all[idx]
    ax_a.plot(
        x_coords,
        medoid_profile,
        marker="o",
        linewidth=2.2,
        markersize=6,
        color=COLORS[k],
        label=f"Persona {k + 1} (#{idx})",
    )

ax_a.set_xticks(x_coords)
ax_a.set_xticklabels(feature_labels, fontsize=9.5)
ax_a.set_ylabel("Normalized Feature Value", fontsize=10)
ax_a.set_title("Medoid Personas Across Attributes", fontsize=11, pad=8)
ax_a.set_ylim(-0.08, 1.42)
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

# Panel (b): 2D MDS semantic space of Gower distances
ax_b = axes[1]
for k in range(3):
    mask = model.labels_ == k
    ax_b.scatter(
        coords_2d[mask, 0],
        coords_2d[mask, 1],
        c=COLORS[k],
        s=45,
        alpha=0.7,
        edgecolors="none",
        label=f"Segment {k + 1}",
    )

med_coords = coords_2d[model.medoid_indices_]
ax_b.scatter(
    med_coords[:, 0],
    med_coords[:, 1],
    marker="*",
    s=220,
    c="gold",
    edgecolors="black",
    linewidths=1.2,
    label="Medoid Personas",
    zorder=5,
)

ax_b.set_title("2D Projection of Gower Metric Space", fontsize=11, pad=8)
ax_b.set_xticks([])
ax_b.set_yticks([])
ax_b.set_ylim(-0.45, 0.55)
ax_b.legend(
    loc="upper center",
    ncol=2,
    frameon=True,
    facecolor="white",
    framealpha=0.92,
    edgecolor="#e0e0e0",
    fontsize=8.5,
)
for spine in ax_b.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

# Panel (c): Real Persona vs. Blurry K-Means Mean on discrete features
ax_c = axes[2]
# Contrast discrete category resolution: FastCLARANS picks an exact channel,
# whereas one-hot K-Means produces fractional channel probabilities.
bar_categories = ["Branch", "Online", "Mobile"]
bar_x = np.arange(len(bar_categories))
bar_width = 0.35

# Take Segment 0 (Mobile Professionals) as an example
seg0_mask = model.labels_ == 0
k_means_channel_means = [
    np.mean(channels[seg0_mask] == 0),
    np.mean(channels[seg0_mask] == 1),
    np.mean(channels[seg0_mask] == 2),
]
medoid_0_idx = model.medoid_indices_[0]
medoid_channel = channels[medoid_0_idx]
medoid_one_hot = [float(medoid_channel == c) for c in range(3)]

ax_c.bar(
    bar_x - bar_width / 2,
    medoid_one_hot,
    width=bar_width,
    color="#2b5c8f",
    label="FastCLARANS Medoid",
    alpha=0.85,
)
ax_c.bar(
    bar_x + bar_width / 2,
    k_means_channel_means,
    width=bar_width,
    color="#d95f02",
    label="K-Means Arithmetic Mean",
    alpha=0.85,
)

ax_c.set_xticks(bar_x)
ax_c.set_xticklabels(bar_categories, fontsize=10)
ax_c.set_ylabel("Channel Value / Probability", fontsize=10)
ax_c.set_title("Concrete Category vs. Fractional Mean", fontsize=11, pad=8)
ax_c.set_ylim(0, 1.35)
ax_c.legend(
    loc="upper right",
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
