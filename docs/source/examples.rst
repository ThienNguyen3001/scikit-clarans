Examples
========

The library provides runnable examples covering basic clustering workflows, custom distance metrics, cluster geometries, and scikit-learn pipeline integration.

For rendered output with generated plots and downloadable Jupyter notebooks (``.ipynb``), visit the :ref:`examples-gallery`.

Overview of examples
--------------------

Quick start and inference:
  * :doc:`auto_examples/plot_quick_start`: Fit a model on Gaussian blobs and inspect cluster medoids.
  * :doc:`auto_examples/plot_predict_new_data`: Assign unseen samples to existing cluster centers.
  * :doc:`auto_examples/plot_transform_data`: Transform observations into distance-to-medoid representations.

Initialization strategies:
  * :doc:`auto_examples/plot_initialization_comparison`: Compare ``k-medoids++``, ``random``, ``build``, and ``heuristic`` seeding.
  * :doc:`auto_examples/plot_custom_init_centers`: Provide user-specified candidate coordinates for domain-guided seeding.

Metrics and data representations:
  * :doc:`auto_examples/plot_metrics_demo`: Evaluate clustering results using Euclidean, Manhattan, and Cosine metrics.
  * :doc:`auto_examples/plot_sparse_input`: Cluster CSR sparse feature matrices directly.
  * :doc:`auto_examples/plot_precomputed_distances`: Cluster datasets using precomputed pairwise distance matrices.

Cluster geometry and diagnostics:
  * :doc:`auto_examples/plot_clusters_blobs`: Medoid identification on isotropic Gaussian clusters.
  * :doc:`auto_examples/plot_clusters_moons`: Behavior on non-convex interleaving moons.
  * :doc:`auto_examples/plot_clusters_anisotropic`: Performance on elongated, anisotropic cluster shapes.
  * :doc:`auto_examples/plot_outlier_robustness`: Compare medoid resistance against centroids in the presence of extreme values.
  * :doc:`auto_examples/plot_silhouette_vs_k`: Select optimal cluster count :math:`k` using silhouette scores.

Performance and benchmarks:
  * :doc:`auto_examples/plot_clarans_vs_fastclarans`: Compare runtime, swap counts, and memory between CLARANS and FastCLARANS.
  * :doc:`auto_examples/plot_cost_evaluation_strategy`: Compare runtime and solution equivalence between delta caching and brute-force evaluation.
  * :doc:`auto_examples/plot_search_trajectory`: Monitor search progress across restarts and track candidate evaluations and swaps.
  * :doc:`auto_examples/plot_runtime_scaling`: Measure execution time scaling across sample sizes.
  * :doc:`auto_examples/plot_parameter_sensitivity`: Analyze trade-offs across ``num_local`` and ``max_neighbors``.
  * :doc:`auto_examples/plot_pipeline_gridsearch`: Integrate estimators into ``scikit-learn`` ``GridSearchCV`` pipelines.

Interactive execution
---------------------

All scripts are available in the `examples directory on GitHub <https://github.com/ThienNguyen3001/scikit-clarans/tree/main/examples>`_.

An interactive notebook is available on Google Colab:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1JdgVaZcbS1uwY7kPQZM8DtX97R9ga31d?usp=sharing
   :alt: Open In Colab