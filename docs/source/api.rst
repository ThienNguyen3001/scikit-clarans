API Reference
=============

Detailed documentation for estimators, initialization strategies, and utility functions in ``scikit-clarans``.

Estimators
----------

CLARANS
^^^^^^^

The classic randomized search k-medoids estimator (Ng & Han, 2002).

.. autoclass:: clarans.CLARANS
   :members:
   :inherited-members:
   :show-inheritance:

FastCLARANS
^^^^^^^^^^^

The accelerated k-medoids estimator using FastPAM1 simultaneous delta updates (Schubert & Rousseeuw, 2021).

.. autoclass:: clarans.FastCLARANS
   :members:
   :inherited-members:
   :show-inheritance:

Helper Modules
--------------

Initialization
^^^^^^^^^^^^^^

Internal strategies for seeding medoids (``k-medoids++``, ``build``, ``heuristic``, ``random``).

.. automodule:: clarans.initialization
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
