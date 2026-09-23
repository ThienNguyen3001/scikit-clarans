Examples for scikit-clarans
===========================

This folder contains runnable example scripts showcasing different aspects of
using the CLARANS implementation.

How to run
----------
From the repository root, run any example with::

    python examples/plot_quick_start.py

List of Gallery Examples (Sphinx-Gallery convention: `plot_*.py`)
-----------------------------------------------------------------

### 1. Basic Clustering & Geometries
- `plot_quick_start.py`: Minimalist 2D clustering and medoids plotting.
- `plot_clusters_blobs.py`: CLARANS on synthetic isotropic Gaussian blobs.
- `plot_clusters_moons.py`: Clustering on non-convex interleaving moons.
- `plot_clusters_anisotropic.py`: Clustering on linearly transformed anisotropic clusters.

### 2. Algorithms & Benchmarks
- `plot_clarans_vs_fastclarans.py`: Side-by-side convergence speed and inertia benchmark.
- `plot_comparison_clustering.py`: Direct visual comparison between CLARANS, FastCLARANS, and K-Means.
- `plot_outlier_robustness.py`: Robustness against extreme anomalies compared to K-Means.
- `plot_runtime_scaling.py`: Empirical execution scaling vs. dataset size N.
- `plot_parameter_sensitivity.py`: Heatmap of solution cost and runtime across parameter grids.

### 3. Initialization & Validation
- `plot_initialization_comparison.py`: Visual comparison of seeding methods (Random, k-medoids++, Heuristic, BUILD).
- `plot_silhouette_vs_k.py`: Silhouette analysis across number of clusters k.

### 4. Advanced Ecosystem Integration
- `plot_metrics_demo.py`: Impact of distance metrics and `metric_params` (Euclidean, Manhattan, Minkowski p=3, Mahalanobis).
- `plot_sparse_input.py`: Clustering on SciPy CSR/CSC sparse matrices.
- `plot_pipeline_gridsearch.py`: Model selection and hyperparameter tuning (k, metric_params) with GridSearchCV.
- `plot_predict_new_data.py`: Assigning unseen query samples to nearest fitted medoids.
- `plot_custom_init_centers.py`: Supplying user-defined domain coordinates as initial seeds.
- `plot_transform_data.py`: Using CLARANS as a feature transformer (cluster-distance space).
- `plot_precomputed_distances.py`: Clustering using precomputed pairwise distance matrices.

Notes
-----
Some examples produce plots using matplotlib and will open interactive windows when
run in an environment that supports it. To run in headless environments (CI, servers),
set the backend::

    import matplotlib
    matplotlib.use("Agg")

If you want an interactive demo, open the [Google Colab Notebook](https://colab.research.google.com/drive/1JdgVaZcbS1uwY7kPQZM8DtX97R9ga31d?usp=sharing)
to run and experiment with `scikit-clarans` directly in your browser.
