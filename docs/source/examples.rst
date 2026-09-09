Examples
========

This page contains runnable examples demonstrating different ways to use
`scikit-clarans` and integrations with popular Python tooling.

Quickstart
----------

A minimal quickstart example:

.. literalinclude:: ../../examples/01_quick_start.py
   :language: python

Compare initializations
-----------------------

Use this script to compare initialization strategies and runtimes:

.. literalinclude:: ../../examples/02_compare_initializations.py
   :language: python

Different distance metrics
--------------------------

See how different metrics affect the clustering result:

.. literalinclude:: ../../examples/03_metrics_demo.py
   :language: python

Using sparse inputs
-------------------

.. literalinclude:: ../../examples/04_sparse_input.py
   :language: python

Grid-search and pipelines
-------------------------

.. literalinclude:: ../../examples/05_pipeline_gridsearch.py
   :language: python

Predicting on new data
----------------------

Assign unseen samples to their nearest fitted cluster medoid:

.. literalinclude:: ../../examples/06_predict_new_data.py
   :language: python

Custom initial medoids
----------------------

Provide user-specified candidate centers for domain-guided seeding:

.. literalinclude:: ../../examples/07_custom_init_centers.py
   :language: python

Performance tuning
------------------

Observe trade-offs between ``numlocal``, ``maxneighbor``, runtime, and solution cost:

.. literalinclude:: ../../examples/08_performance_tuning.py
   :language: python

CLARANS vs FastCLARANS
----------------------

Benchmark runtime, memory, and solution quality between the two algorithms:

.. literalinclude:: ../../examples/09_compare_fastclarans_clarans.py
   :language: python

Feature transformation
----------------------

Transform datasets into a cluster-distance representation using ``transform()``:

.. literalinclude:: ../../examples/10_transform_data.py
   :language: python

Precomputed distance matrices
-----------------------------

Cluster non-vector or graph datasets using a precomputed pairwise distance matrix (``metric="precomputed"``):

.. literalinclude:: ../../examples/11_precomputed_distances.py
   :language: python

Interactive Demo & Additional Resources
---------------------------------------

Additional runnable scripts are available in the `examples directory <https://github.com/ThienNguyen3001/scikit-clarans/tree/main/examples>`_.

You can also run and modify interactive experiments directly in Google Colab:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1JdgVaZcbS1uwY7kPQZM8DtX97R9ga31d?usp=sharing
   :alt: Open In Colab