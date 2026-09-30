.. scikit-clarans documentation master file, created by
   sphinx-quickstart on Sat Jan 17 12:05:36 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scikit-clarans
==============

``scikit-clarans`` implements the CLARANS (Clustering Large Applications based on RANdomized Search) and FastCLARANS algorithms for Python using the scikit-learn estimator interface.

Key characteristics:

* Scikit-learn compatibility: estimators implement ``fit``, ``predict``, and ``transform`` as standard clusterers.
* Low memory footprint: computes distances dynamically with :math:`O(n)` memory instead of allocating an :math:`O(n^2)` pairwise matrix.
* C acceleration: core swap evaluation runs in Cython with a pure Python fallback.
* Flexible metrics: supports dense arrays via SciPy, sparse matrices via scikit-learn, and custom callables.

.. note::
   This library is designed for coursework, algorithm study, and research prototyping. It scales well to tens of thousands of samples on a single machine, but it is not intended for distributed data systems.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 2
   :caption: Example Gallery

   auto_examples/index

.. toctree::
   :maxdepth: 3
   :caption: Project

   project
   contributing
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

