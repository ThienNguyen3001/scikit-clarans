"""
==========================================================
Comparing Different Clustering Algorithms on Toy Datasets
==========================================================

This example showcases characteristics of different clustering algorithms on
synthetic 2D datasets with varied geometry, evaluating 11 standard clustering
estimators from scikit-learn alongside :class:`~clarans.CLARANS` and
:class:`~clarans.FastCLARANS`.

The script produces two figures:

1. **Full Comparison Grid**: Direct side-by-side benchmark across all 13
   algorithms on 6 toy datasets.
2. **Focused K-Medoids Comparison**: High-resolution view of CLARANS vs.
   FastCLARANS highlighting true medoids (marked with ``x``) and execution times.
"""

# Authors: The scikit-learn developers
# Modified by: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com> (adapted for scikit-clarans)
# SPDX-License-Identifier: BSD-3-Clause

import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from clarans import CLARANS, FastCLARANS

# Generate synthetic datasets with distinct geometries
n_samples = 500
seed = 30
rng = np.random.RandomState(seed)

noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
no_structure = (rng.rand(n_samples, 2), None)

# Anisotropically distributed clusters
random_state = 170
X_aniso_raw, y_aniso = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X_aniso_raw, transformation)
aniso = (X_aniso, y_aniso)

# Blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

datasets_list = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (no_structure, {}),
]

dataset_names = [
    "Circles",
    "Moons",
    "Varied Variances",
    "Anisotropic",
    "Blobs",
    "No Structure",
]

# Set up figures:
# Figure 1: Full 13-algorithm comparison grid
fig1 = plt.figure(figsize=(13 * 2 + 1, 13))
fig1.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01)

# Figure 2: Focused 2-column view of CLARANS and FastCLARANS
fig2 = plt.figure(figsize=(6.5, 14.0))
fig2.subplots_adjust(left=0.18, right=0.98, bottom=0.01, top=0.96, wspace=0.08, hspace=0.12)

for i_dataset, (dataset, algo_params) in enumerate(datasets_list):
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # Estimate bandwidth for MeanShift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # Connectivity matrix for structured Ward and Agglomerative clustering
    connectivity = kneighbors_graph(X, n_neighbors=params["n_neighbors"], include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Instantiate clustering estimators
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(
        n_clusters=params["n_clusters"],
        random_state=params["random_state"],
    )
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=params["random_state"],
    )
    dbscan = cluster.DBSCAN(eps=params["eps"])
    hdbscan = cluster.HDBSCAN(
        min_samples=params["hdbscan_min_samples"],
        min_cluster_size=params["hdbscan_min_cluster_size"],
        allow_single_cluster=params["allow_single_cluster"],
        copy=True,
    )
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )
    affinity_propagation = cluster.AffinityPropagation(
        damping=params["damping"],
        preference=params["preference"],
        random_state=params["random_state"],
    )
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        metric="cityblock",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"],
        covariance_type="full",
        random_state=params["random_state"],
    )

    clarans = CLARANS(
        n_clusters=params["n_clusters"],
        random_state=params["random_state"],
    )
    fast_clarans = FastCLARANS(
        n_clusters=params["n_clusters"],
        random_state=params["random_state"],
    )

    clustering_algorithms = (
        ("MiniBatch\nKMeans", two_means),
        ("Affinity\nPropagation", affinity_propagation),
        ("MeanShift", ms),
        ("Spectral\nClustering", spectral),
        ("Ward", ward),
        ("Agglomerative\nClustering", average_linkage),
        ("DBSCAN", dbscan),
        ("HDBSCAN", hdbscan),
        ("OPTICS", optics),
        ("BIRCH", birch),
        ("Gaussian\nMixture", gmm),
        ("CLARANS", clarans),
        ("FastCLARANS", fast_clarans),
    )

    for i_algo, (name, algorithm) in enumerate(clustering_algorithms):
        t0 = time.perf_counter()

        # Catch warnings related to kneighbors_graph connectivity
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                "connectivity matrix is [0-9]{1,2}"
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                " may not work as expected.",
                category=UserWarning,
            )
            algorithm.fit(X)

        elapsed = time.perf_counter() - t0
        time_str = f"{elapsed:.2f}s".lstrip("0")

        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        colors = np.append(colors, ["#000000"])

        # ------------------------------------------------------------------
        # Figure 1: Full 13-algorithm comparison grid
        # ------------------------------------------------------------------
        plot_num_fig1 = i_dataset * len(clustering_algorithms) + i_algo + 1
        ax1 = fig1.add_subplot(len(datasets_list), len(clustering_algorithms), plot_num_fig1)
        if i_dataset == 0:
            ax1.set_title(name, size=14)

        ax1.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylim(-2.5, 2.5)
        ax1.set_xticks(())
        ax1.set_yticks(())
        ax1.text(
            0.99,
            0.01,
            time_str,
            transform=ax1.transAxes,
            size=15,
            horizontalalignment="right",
        )

        # ------------------------------------------------------------------
        # Figure 2: Focused CLARANS and FastCLARANS side-by-side view
        # ------------------------------------------------------------------
        if name in ("CLARANS", "FastCLARANS"):
            col_idx = 0 if name == "CLARANS" else 1
            ax2 = fig2.add_subplot(len(datasets_list), 2, i_dataset * 2 + col_idx + 1)
            if i_dataset == 0:
                ax2.set_title(name, size=14, fontweight="bold")
            if col_idx == 0:
                ax2.set_ylabel(
                    dataset_names[i_dataset],
                    size=12,
                    fontweight="bold",
                    labelpad=10,
                )

            ax2.scatter(X[:, 0], X[:, 1], s=14, color=colors[y_pred], alpha=0.8)

            centers = getattr(algorithm, "cluster_centers_", None)
            if centers is not None:
                ax2.scatter(
                    centers[:, 0],
                    centers[:, 1],
                    marker="x",
                    s=80,
                    c="black",
                    linewidths=2.0,
                    zorder=10,
                )

            ax2.set_xlim(-2.5, 2.5)
            ax2.set_ylim(-2.5, 2.5)
            ax2.set_xticks(())
            ax2.set_yticks(())
            for spine in ax2.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(0.8)

            ax2.text(
                0.97,
                0.03,
                time_str,
                transform=ax2.transAxes,
                size=12,
                fontweight="bold",
                horizontalalignment="right",
            )

plt.show()
