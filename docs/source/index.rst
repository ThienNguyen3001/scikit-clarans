.. scikit-clarans documentation master file, created by
   sphinx-quickstart on Sat Jan 17 12:05:36 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scikit-clarans Documentation
============================

Welcome to the documentation for **scikit-clarans**, a specialized clustering library for Python.

**scikit-clarans** brings the power of the CLARANS (Clustering Large Applications based on RANdomized Search) algorithm to the **scikit-learn** ecosystem. It is designed to be:

*   **High Performance & Memory-Efficient**: Cython-accelerated C-extensions (with pure Python fallback) paired with :math:`O(n)` memory overhead, substantially more scalable than classic :math:`O(n^2)` PAM.
*   **Compatible**: A drop-in replacement for scikit-learn clusterers efficiently implementing ``fit``, ``predict``, and more.
*   **Flexible & Fast**: Cascading distance engine supporting dense arrays (SciPy C-kernel), sparse matrices (Scikit-Learn DistanceMetric), and custom metrics.

.. note::
   **Educational & Academic Research Scope**

   ``scikit-clarans`` is developed primarily for **learning, algorithmic exploration, and small-to-medium academic research**. While its Cython-accelerated core and :math:`O(n)` memory footprint provide significant speedups over standard PAM, it is designed for clean, accessible prototyping rather than massive distributed enterprise Big Data pipelines (:math:`N \gg 10^5`).

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

