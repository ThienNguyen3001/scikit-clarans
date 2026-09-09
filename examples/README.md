Examples for scikit-clarans
===========================

This folder contains runnable example scripts showcasing different aspects of
using the CLARANS implementation.

How to run
----------
From the repository root, run any example with::

    python examples/01_quick_start.py

List of examples
----------------

- 01_quick_start.py: Simple clustering and visualization in 2D.
- 02_compare_initializations.py: Compare initialization strategies and their cost/runtime.
- 03_metrics_demo.py: Demonstrate different distance metrics (euclidean, manhattan, cosine).
- 04_sparse_input.py: Use CLARANS on CSR sparse matrices.
- 05_pipeline_gridsearch.py: Use CLARANS in a scikit-learn Pipeline and tune with GridSearchCV + silhouette score.
- 06_predict_new_data.py: Predict cluster labels for new data points.
- 07_custom_init_centers.py: Pass an array-like init to CLARANS.
- 08_performance_tuning.py: Demonstrate runtime / quality trade-offs when tuning parameters.
- 09_compare_fastclarans_clarans.py: Compare CLARANS vs. FastCLARANS (FastPAM1 optimization with on-the-fly distances).
- 10_transform_data.py: Demonstrate CLARANS transform method for distance-to-medoid features.
- 11_precomputed_distances.py: Cluster using precomputed distance matrices (metric="precomputed").

Notes
-----
Some examples produce plots using matplotlib and will open interactive windows when
run in an environment that supports it. To run in headless environments (CI, servers),
set the backend::

    import matplotlib
    matplotlib.use("Agg")

If you want an interactive demo, open the [Google Colab Notebook](https://colab.research.google.com/drive/1JdgVaZcbS1uwY7kPQZM8DtX97R9ga31d?usp=sharing)
to run and experiment with `scikit-clarans` directly in your browser.
