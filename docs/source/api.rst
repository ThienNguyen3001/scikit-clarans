API Reference
=============

Estimators, initialization strategies, and utility functions.

Estimators
----------

CLARANS
^^^^^^^

Randomized search k-medoids estimator (Ng & Han, 2002).

.. autoclass:: clarans.CLARANS
   :members:
   :inherited-members:
   :show-inheritance:

FastCLARANS
^^^^^^^^^^^

FastPAM1 k-medoids estimator (Schubert & Rousseeuw, 2021).

.. autoclass:: clarans.FastCLARANS
   :members:
   :inherited-members:
   :show-inheritance:

Helper Modules
--------------

Initialization
^^^^^^^^^^^^^^

Medoid initialization strategies: ``k-medoids++``, ``build``, ``heuristic``, and random sampling.

.. automodule:: clarans._initialization
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
^^^^^^^^^

Distance metrics, objective evaluations, and validation helpers.

.. automodule:: clarans.utils
   :members:
   :undoc-members:
   :show-inheritance:
