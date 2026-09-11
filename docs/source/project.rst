Project
=======

``scikit-clarans`` is a scikit-learn compatible Python library for scalable k-medoids clustering using the CLARANS and FastCLARANS algorithms. The documentation contains usage guides, practical tutorials, an API reference, and a gallery of runnable examples.

Citation
--------

If you use ``scikit-clarans`` in your software or research, please cite:

.. code-block:: bibtex

    @software{scikit_clarans,
      author       = {Nguyen, Ngoc Thien},
      title        = {scikit-clarans: A Python Library for CLARANS Clustering},
      year         = {2026},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.18366801},
      url          = {https://github.com/ThienNguyen3001/scikit-clarans}
    }

Academic References
-------------------

The algorithms and heuristics implemented in this library originate from the following research:

* **CLARANS:**
  Ng, R. T., & Han, J. (2002). *CLARANS: A method for clustering objects for spatial data mining.*
  IEEE Transactions on Knowledge and Data Engineering, 14(5), 1003-1016.
  `doi:10.1109/TKDE.2002.1033770 <https://doi.org/10.1109/TKDE.2002.1033770>`_

* **FastCLARANS & FastPAM1:**
  Schubert, E., & Rousseeuw, P. J. (2021). *Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms.*
  Information Systems, 101, 101804.
  `doi:10.1016/j.is.2021.101804 <https://doi.org/10.1016/j.is.2021.101804>`_

* **Seeding Methods:**

  * Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding.* SODA '07.
  * Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis.* John Wiley & Sons.
  * Initialization implementations (``'k-medoids++'``, ``'heuristic'``, ``'build'``) adapted from `scikit-learn-extra KMedoids <https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html>`_.

Resources
---------

* GitHub repository: https://github.com/ThienNguyen3001/scikit-clarans
* PyPI: https://pypi.org/project/scikit-clarans/
* Documentation: :doc:`index`
* Examples: :doc:`examples`
* Contributing guidelines: :doc:`contributing`
* License: :doc:`license`
