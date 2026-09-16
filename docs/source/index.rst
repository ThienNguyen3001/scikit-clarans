.. scikit-clarans documentation master file, created by
   sphinx-quickstart on Sat Jan 17 12:05:36 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scikit-clarans Documentation
============================

Welcome to the documentation for **scikit-clarans**, a specialized clustering library for Python.

**scikit-clarans** brings the power of the CLARANS (Clustering Large Applications based on RANdomized Search) algorithm to the **scikit-learn** ecosystem. It is designed to be:

*   **Accessible & Memory-Efficient**: Pure Python/NumPy implementation with :math:`O(n)` memory overhead, substantially more scalable than classic :math:`O(n^2)` PAM.
*   **Compatible**: A drop-in replacement for scikit-learn clusterers efficiently implementing ``fit``, ``predict``, and more.
*   **Flexible**: Supports customizable initialization strategies and distance metrics.

.. note::
   **Educational & Academic Research Scope**

   ``scikit-clarans`` is developed primarily for **learning, algorithmic exploration, and small-to-medium academic research**. While it avoids the heavy :math:`O(n^2)` memory consumption of standard PAM, the current pure Python implementation is **not yet optimized for large-scale enterprise Big Data pipelines** (:math:`N \gg 10^5`).

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
   :maxdepth: 3
   :caption: Gallery

   gallery/2d_clustering
   gallery/comparison
   gallery/outliers
   gallery/initializations
   gallery/quality_vs_k
   gallery/performance

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

