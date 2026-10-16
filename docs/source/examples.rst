Examples
========

The library provides runnable examples covering basic clustering workflows, custom distance metrics, cluster geometries, and scikit-learn pipeline integration.

For rendered output with generated plots and downloadable Jupyter notebooks (``.ipynb``), visit the :ref:`examples-gallery`.

Overview of examples
--------------------

Cluster geometries:
  * :doc:`auto_examples/cluster_geometry/plot_clusters_blobs`: Medoid identification on isotropic Gaussian clusters.
  * :doc:`auto_examples/cluster_geometry/plot_clusters_moons`: Behavior on non-convex interleaving moons.
  * :doc:`auto_examples/cluster_geometry/plot_clusters_anisotropic`: Performance on elongated, anisotropic cluster shapes.

Distance metrics and representations:
  * :doc:`auto_examples/distance_metrics/plot_metrics_demo`: Evaluate clustering results using Euclidean, Manhattan, and Cosine metrics.
  * :doc:`auto_examples/distance_metrics/plot_sparse_input`: Cluster CSR sparse feature matrices directly.
  * :doc:`auto_examples/distance_metrics/plot_precomputed_distances`: Cluster datasets using precomputed pairwise distance matrices.
  * :doc:`auto_examples/distance_metrics/plot_graph_clustering`: Cluster network nodes with shortest-path distances.
  * :doc:`auto_examples/distance_metrics/plot_nan_data_clustering`: Cluster incomplete data with missing values (NaN) using ``nan_euclidean``.
  * :doc:`auto_examples/distance_metrics/plot_custom_callable_metric`: Cluster geographic coordinates using a custom Haversine distance callable.
  * :doc:`auto_examples/distance_metrics/plot_asymmetric_distances`: Cluster directed networks on asymmetric precomputed distance matrices.
  * :doc:`auto_examples/distance_metrics/plot_string_clustering`: Cluster non-vector text sequences using Levenshtein edit distance.
  * :doc:`auto_examples/distance_metrics/plot_time_series_dtw`: Cluster temporal signals with Dynamic Time Warping (DTW) distance matrices.
  * :doc:`auto_examples/distance_metrics/plot_mixed_data_gower`: Cluster mixed continuous, binary, and categorical data using Gower's dissimilarity.

Seeding and validation:
  * :doc:`auto_examples/initialization/plot_initialization_comparison`: Compare ``k-medoids++``, ``random``, ``build``, and ``heuristic`` seeding.
  * :doc:`auto_examples/initialization/plot_custom_init_centers`: Provide user-specified candidate coordinates for domain-guided seeding.
  * :doc:`auto_examples/initialization/plot_silhouette_vs_k`: Select optimal cluster count :math:`k` using silhouette scores.

Inference and pipelines:
  * :doc:`auto_examples/inference/plot_predict_new_data`: Assign unseen samples to existing cluster centers.
  * :doc:`auto_examples/inference/plot_transform_data`: Transform observations into distance-to-medoid representations.
  * :doc:`auto_examples/inference/plot_precomputed_inference`: Accelerate precomputed inference using compact (n_test, k) distance submatrices.
  * :doc:`auto_examples/inference/plot_pipeline_gridsearch`: Integrate estimators into ``scikit-learn`` ``GridSearchCV`` pipelines.

Robustness and anomaly detection:
  * :doc:`auto_examples/robustness/plot_outlier_robustness`: Compare medoid resistance against centroids in the presence of extreme values.
  * :doc:`auto_examples/robustness/plot_anomaly_detection`: Unsupervised outlier and anomaly detection via distance-to-medoid scoring.

Benchmarks and convergence:
  * :doc:`auto_examples/benchmarks/plot_clarans_vs_fastclarans`: Compare runtime, swap counts, and memory between CLARANS and FastCLARANS.
  * :doc:`auto_examples/benchmarks/plot_cost_evaluation_strategy`: Compare runtime and solution equivalence between delta caching and brute-force evaluation.
  * :doc:`auto_examples/benchmarks/plot_search_trajectory`: Monitor search progress across restarts and track candidate evaluations and swaps.
  * :doc:`auto_examples/benchmarks/plot_comparison_clustering`: Direct visual comparison between CLARANS, FastCLARANS, and K-Means.
  * :doc:`auto_examples/benchmarks/plot_runtime_scaling`: Measure execution time scaling across sample sizes.
  * :doc:`auto_examples/benchmarks/plot_parameter_sensitivity`: Analyze trade-offs across ``num_local`` and ``max_neighbors``.

Interactive execution
---------------------

All scripts are available in the `examples directory on GitHub <https://github.com/ThienNguyen3001/scikit-clarans/tree/main/examples>`_.

An interactive notebook is available on Google Colab:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1JdgVaZcbS1uwY7kPQZM8DtX97R9ga31d?usp=sharing
   :alt: Open In Colab