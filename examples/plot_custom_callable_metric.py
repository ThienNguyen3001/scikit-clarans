"""
=================================================================
Custom Callable Distance Metric: Geospatial Haversine Clustering
=================================================================

This example demonstrates how CLARANS and FastCLARANS natively accept any
user-defined distance function (``metric=callable``). We apply a custom
Haversine distance function to cluster geographic coordinates (latitude and
longitude) on the Earth's spherical surface.

Unlike K-Means, which cannot accept custom distance callables and generates
fictitious synthetic centroids (which might fall in oceans or inaccessible
locations), CLARANS identifies actual physical locations from the dataset
as regional logistics distribution hubs (medoids).

.. note::

    **Haversine Great-Circle Distance:**
    The Haversine formula calculates the shortest distance over the Earth's
    spherical surface between two points :math:`(lat_1, lon_1)` and
    :math:`(lat_2, lon_2)`:

    .. math::

        d = 2 R \\arcsin \\left(\\sqrt{\\sin^2\\left(\\frac{\\Delta lat}{2}\\right) +
        \\cos(lat_1) \\cos(lat_2) \\sin^2\\left(\\frac{\\Delta lon}{2}\\right)}\\right)

    where :math:`R \\approx 6371.0` km is the mean radius of Earth, and coordinates
    are converted to radians.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from clarans import FastCLARANS

EARTH_RADIUS_KM = 6371.0
COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]


def haversine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute great-circle distance between two (lat, lon) coordinates in kilometers."""
    lat1, lon1 = np.radians(u[0]), np.radians(u[1])
    lat2, lon2 = np.radians(v[0]), np.radians(v[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(EARTH_RADIUS_KM * c)


# 1. Synthesize customer delivery locations around 4 regional metro clusters (Lat, Lon)
rng = np.random.RandomState(42)
hubs = [
    (10.8231, 106.6297),  # Region 1 (South)
    (16.0544, 108.2022),  # Region 2 (Central)
    (21.0285, 105.8542),  # Region 3 (North)
    (10.0452, 105.7469),  # Region 4 (Delta)
]

coords_list = []
for lat, lon in hubs:
    cluster_lats = rng.normal(loc=lat, scale=0.45, size=60)
    cluster_lons = rng.normal(loc=lon, scale=0.45, size=60)
    coords_list.append(np.column_stack([cluster_lats, cluster_lons]))

X = np.vstack(coords_list)

# 2. Fit FastCLARANS with our custom Haversine callable metric
model = FastCLARANS(
    n_clusters=4,
    num_local=3,
    metric=haversine_distance,
    random_state=42,
)
model.fit(X)

print(f"Optimal Medoid Indices:        {model.medoid_indices_}")
print(f"Total Network Haversine Cost:  {model.inertia_:,.1f} km")

# 3. Visualization
fig, ax = plt.subplots(figsize=(7.5, 6.0))

for k in range(4):
    mask = model.labels_ == k
    ax.scatter(
        X[mask, 1],
        X[mask, 0],
        c=COLORS[k],
        s=30,
        alpha=0.7,
        edgecolors="none",
        label=f"Region {k + 1}",
    )

medoids = model.cluster_centers_
ax.scatter(
    medoids[:, 1],
    medoids[:, 0],
    marker="*",
    s=220,
    c="gold",
    edgecolors="black",
    linewidths=1.5,
    label="Selected Logistics Hubs",
    zorder=5,
)

# Connect delivery points to their assigned regional hub with faint lines
for i in range(len(X)):
    hub = medoids[model.labels_[i]]
    ax.plot(
        [X[i, 1], hub[1]],
        [X[i, 0], hub[0]],
        color=COLORS[model.labels_[i]],
        alpha=0.12,
        linewidth=0.8,
        zorder=1,
    )

ax.set_title(
    f"Geospatial Clustering with Custom Haversine Metric\n"
    f"Total Network Transport Cost = {model.inertia_:,.1f} km",
    fontsize=11,
    pad=10,
)
ax.set_xlabel("Longitude (Degrees)", fontsize=9.5)
ax.set_ylabel("Latitude (Degrees)", fontsize=9.5)
for spine in ax.spines.values():
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)

ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
plt.tight_layout()
plt.show()
