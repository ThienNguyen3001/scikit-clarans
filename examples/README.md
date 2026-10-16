Examples for scikit-clarans
===========================

This folder contains runnable example scripts showcasing different aspects of
using the CLARANS implementation.

How to run
----------
From the repository root, run any example with::

    python examples/cluster_geometry/plot_clusters_blobs.py

List of Gallery Examples by Category
------------------------------------

### 1. Cluster Geometries (`examples/cluster_geometry/`)
- `plot_clusters_blobs.py`: CLARANS on synthetic isotropic Gaussian blobs.
- `plot_clusters_moons.py`: Clustering on non-convex interleaving moons.
- `plot_clusters_anisotropic.py`: Clustering on linearly transformed anisotropic clusters.

### 2. Distance Metrics & Representations (`examples/distance_metrics/`)
- `plot_metrics_demo.py`: Impact of distance metrics and `metric_params` (Euclidean, Manhattan, Minkowski p=3, Mahalanobis).
- `plot_sparse_input.py`: Clustering on SciPy CSR/CSC sparse matrices.
- `plot_precomputed_distances.py`: Clustering using precomputed pairwise distance matrices.
- `plot_graph_clustering.py`: Graph community clustering with precomputed shortest-path distances.
- `plot_nan_data_clustering.py`: Handling missing values (NaN) with `metric='nan_euclidean'`.
- `plot_custom_callable_metric.py`: Custom callable distance metric (Haversine geospatial GPS clustering).
- `plot_asymmetric_distances.py`: Clustering on directed / asymmetric distance matrices ($D \neq D^T$).
- `plot_string_clustering.py`: Clustering non-vector text sequences with Levenshtein edit distance.
- `plot_time_series_dtw.py`: Clustering temporal signals with Dynamic Time Warping (DTW) distance matrices.
- `plot_mixed_data_gower.py`: Clustering mixed continuous, binary, and categorical data using Gower's dissimilarity.

### 3. Seeding & Validation (`examples/initialization/`)
- `plot_initialization_comparison.py`: Visual comparison of seeding methods (Random, k-medoids++, Heuristic, BUILD).
- `plot_custom_init_centers.py`: Supplying user-defined domain coordinates as initial seeds.
- `plot_silhouette_vs_k.py`: Silhouette analysis across number of clusters k.

### 4. Inference & Pipelines (`examples/inference/`)
- `plot_predict_new_data.py`: Assigning unseen query samples to nearest fitted medoids.
- `plot_transform_data.py`: Using CLARANS as a feature transformer (cluster-distance space).
- `plot_pipeline_gridsearch.py`: Model selection and hyperparameter tuning (k, metric_params) with GridSearchCV.
- `plot_precomputed_inference.py`: Fast precomputed inference using compact $(n_{\text{test}}, k)$ matrices.

### 5. Robustness & Anomaly Detection (`examples/robustness/`)
- `plot_outlier_robustness.py`: Robustness against extreme anomalies compared to K-Means.
- `plot_anomaly_detection.py`: Unsupervised anomaly detection via distance-to-medoid scoring.

### 6. Benchmarks & Convergence (`examples/benchmarks/`)
- `plot_clarans_vs_fastclarans.py`: Side-by-side convergence speed and inertia benchmark.
- `plot_cost_evaluation_strategy.py`: Delta caching vs. brute-force swap evaluation runtime and equivalence.
- `plot_search_trajectory.py`: Monitoring search progress, restart convergence, and tracking metrics.
- `plot_comparison_clustering.py`: Direct visual comparison between CLARANS, FastCLARANS, and K-Means.
- `plot_runtime_scaling.py`: Empirical execution scaling vs. dataset size N.
- `plot_parameter_sensitivity.py`: Heatmap of solution cost and runtime across parameter grids.

Notes
-----
Some examples produce plots using matplotlib and will open interactive windows when
run in an environment that supports it. To run in headless environments (CI, servers),
set the backend::

    import matplotlib
    matplotlib.use("Agg")

If you want an interactive demo, open the [Google Colab Notebook](https://colab.research.google.com/drive/1JdgVaZcbS1uwY7kPQZM8DtX97R9ga31d?usp=sharing)
to run and experiment with `scikit-clarans` directly in your browser.
