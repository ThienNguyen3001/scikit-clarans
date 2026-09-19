Examples
========

This page contains runnable examples demonstrating different ways to use
`scikit-clarans` and integrations with popular Python tooling.

Quickstart
----------

A minimal quickstart example:

.. literalinclude:: ../../examples/plot_quick_start.py
   :language: python

Compare initializations
-----------------------

Use this script to compare initialization strategies and runtimes:

.. literalinclude:: ../../examples/plot_initialization_comparison.py
   :language: python

Different distance metrics
--------------------------

See how different metrics affect the clustering result:

.. literalinclude:: ../../examples/plot_metrics_demo.py
   :language: python

Using sparse inputs
-------------------

.. literalinclude:: ../../examples/plot_sparse_input.py
   :language: python

Grid-search and pipelines
-------------------------

.. literalinclude:: ../../examples/plot_pipeline_gridsearch.py
   :language: python

Predicting on new data
----------------------

Assign unseen samples to their nearest fitted cluster medoid:

.. literalinclude:: ../../examples/plot_predict_new_data.py
   :language: python

Custom initial medoids
----------------------

Provide user-specified candidate centers for domain-guided seeding:

.. literalinclude:: ../../examples/plot_custom_init_centers.py
   :language: python

Performance tuning
------------------

Observe trade-offs between ``num_local``, ``max_neighbors``, runtime, and solution cost:

.. literalinclude:: ../../examples/plot_parameter_sensitivity.py
   :language: python

CLARANS vs FastCLARANS
----------------------

Benchmark runtime, memory, and solution quality between the two algorithms:

.. literalinclude:: ../../examples/plot_clarans_vs_fastclarans.py
   :language: python

Feature transformation
----------------------

Transform datasets into a cluster-distance representation using ``transform()``:

.. literalinclude:: ../../examples/plot_transform_data.py
   :language: python

Precomputed distance matrices
-----------------------------

Cluster non-vector or graph datasets using a precomputed pairwise distance matrix (``metric="precomputed"``):

.. literalinclude:: ../../examples/plot_precomputed_distances.py
   :language: python

Interactive Demo & Additional Resources
---------------------------------------

Additional runnable scripts are available in the `examples directory <https://github.com/ThienNguyen3001/scikit-clarans/tree/main/examples>`_.

You can also run and modify interactive experiments directly in Google Colab:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1JdgVaZcbS1uwY7kPQZM8DtX97R9ga31d?usp=sharing
   :alt: Open In Colab