User guide
==========

``scikit-clarans`` provides estimators for $k$-medoids clustering. This guide covers algorithm selection, parameter tuning, and search mechanics.

Why use k-medoids?
------------------

Standard $k$-means calculates cluster centers by averaging coordinates in Euclidean space, which creates synthetic centroids. In contrast, $k$-medoids restricts cluster centers to actual observations from the dataset:

* Outlier resistance: Medoids minimize absolute pairwise distances (:math:`\sum d`) rather than squared Euclidean distances (:math:`\sum d^2`), keeping centers stable when the dataset contains extreme values.
* Custom distance metrics: Supports metrics such as Manhattan distance, cosine dissimilarity, or precomputed distance matrices, whereas standard $k$-means requires Euclidean space.
* Interpretability: Every medoid corresponds to a real record in the input data, such as a patient profile, an exemplary document, or a molecule.

.. math::

   \min_{M \subset X, |M|=k} \sum_{i=1}^n \min_{m \in M} d(x_i, m)

.. note::

   In ``scikit-learn``'s ``KMeans``, ``inertia_`` represents the sum of squared Euclidean distances. In ``scikit-clarans``, ``inertia_`` represents the sum of unsquared distances from each sample to its assigned medoid according to the chosen metric.

Quick start
-----------

Clustering with ``CLARANS``:

.. code-block:: python

    from clarans import CLARANS
    from sklearn.datasets import make_blobs

    # 1. Generate sample data
    X, _ = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=42)

    # 2. Instantiate and fit model
    model = CLARANS(
        n_clusters=4,
        num_local=3,
        init='k-medoids++',
        random_state=42
    )
    model.fit(X)

    # 3. Inspect results
    print("Medoid Indices:", model.medoid_indices_)
    print("Medoid Coordinates:\n", model.cluster_centers_)
    print("Labels (first 10):", model.labels_[:10])
    print("Inertia (Total Distance Cost):", model.inertia_)

Choosing an estimator
---------------------

``scikit-clarans`` includes two estimators:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - CLARANS
     - FastCLARANS
   * - Reference
     - Ng & Han (2002)
     - Schubert & Rousseeuw (2021)
   * - Candidate sampling
     - Random (medoid, non-medoid) pairs
     - Non-medoid samples tested against all :math:`k` medoids
   * - Swap evaluation
     - Single swap evaluation (:math:`O(n)` with ``cost_evaluation='delta'``, :math:`O(n \cdot k)` with ``cost_evaluation='brute_force'``)
     - FastPAM1 vectorized delta cost across all :math:`k` medoids
   * - Working memory
     - :math:`O(n)`
     - :math:`O(n)`
   * - Primary use
     - Baseline and algorithm comparison
     - General clustering workflows and benchmarks

.. note::
   Both estimators compute distances dynamically with an :math:`O(n)` memory footprint rather than storing an :math:`O(n^2)` matrix. They run in a single process and typically handle tens of thousands of samples efficiently.

FastCLARANS example:

.. code-block:: python

    from clarans import FastCLARANS

    fast_model = FastCLARANS(n_clusters=4, num_local=3, random_state=42)
    fast_model.fit(X)

Configuration and parameter tuning
----------------------------------

Both estimators share core parameters, with ``CLARANS`` providing an additional ``cost_evaluation`` option:

.. list-table::
   :widths: 18 15 15 52
   :header-rows: 1

   * - Parameter
     - Estimator
     - Default
     - Description
   * - ``n_clusters``
     - Both
     - ``8``
     - Number of clusters (medoids) to find (:math:`k`).
   * - ``num_local``
     - Both
     - ``2``
     - Number of local searches (random restarts). Higher values explore more local minima.
   * - ``max_neighbors``
     - Both
     - ``'auto'``
     - Maximum non-improving neighbors to check per search. Defaults to :math:`\max(250, 1.25\% \times k(n-k))` in CLARANS and :math:`\max(1, \lfloor 250/k \rfloor, 2.5\% \times (n-k))` in FastCLARANS.
   * - ``init``
     - Both
     - ``'k-medoids++'``
     - Initialization strategy (``'k-medoids++'``, ``'random'``, ``'build'``, ``'heuristic'``, or array-like).
   * - ``metric``
     - Both
     - ``'euclidean'``
     - Distance metric to use (e.g., ``'euclidean'``, ``'manhattan'``, ``'cosine'``).
   * - ``metric_params``
     - Both
     - ``None``
     - Additional keyword arguments for the metric function (e.g., ``{"p": 3}`` for Minkowski).
   * - ``cost_evaluation``
     - CLARANS only
     - ``'delta'``
     - Swap evaluation method (``'delta'`` for :math:`O(n)` evaluations using distance caching; ``'brute_force'`` for recalculating total cost in :math:`O(n \cdot k)`).
   * - ``verbose``
     - Both
     - ``0``
     - Verbosity level (``0`` for silent, ``1`` for per-search progress summary, ``>=2`` for per-swap details).

Tuning guidelines
^^^^^^^^^^^^^^^^^

Cost evaluation in CLARANS:
  ``CLARANS`` supports ``cost_evaluation='delta'`` (default) and ``cost_evaluation='brute_force'``. FastCLARANS does not expose this parameter because its FastPAM1 formulation always calculates deltas.
  
  * ``cost_evaluation='delta'`` caches nearest (:math:`d_1`) and second-nearest (:math:`d_2`) medoid distances for all samples. Each candidate swap runs in :math:`O(n \cdot d)` operations without recomputing distances to unchanged medoids. This gives a 1.5x to 2.5x speedup with identical clustering results.
  * ``cost_evaluation='brute_force'`` recalculates total clustering cost from scratch in :math:`O(n \cdot k \cdot d)`, reproducing the original formulation of Ng & Han (2002).

Initialization strategy (``init``):
  * ``'k-medoids++'`` (default): Selects initial medoids probabilistically based on squared distances. Fast and requires :math:`O(n \cdot k)` memory.
  * ``'random'``: Uniform random sampling. Very fast, though it may require increasing ``num_local`` to reach comparable solution quality.
  * ``'build'``: Greedy initialization from classic PAM. Provides good starting points on small datasets, but computes the full pairwise distance matrix (:math:`O(n^2)` time and memory). Avoid for datasets larger than 5,000 samples.
  * ``'heuristic'``: Selects the :math:`k` samples with the smallest sum of distances to all other points (:math:`O(n^2)` time and memory). Avoid for datasets larger than 5,000 samples.
  * Array-like: User-provided array of initial medoid coordinates or indices.

Number of restarts (``num_local``):
  For randomized initialization (``'k-medoids++'`` or ``'random'``), increasing ``num_local`` to 3 to 5 helps find better local minima when solutions vary across runs.
  
  .. note::
     Deterministic initialization methods such as ``'build'``, ``'heuristic'``, or explicit coordinate arrays always produce the same starting points. With deterministic seeding, set ``num_local=1`` to avoid repeated identical searches.

Neighbor sampling (``max_neighbors``):
  The default ``max_neighbors='auto'`` scales with the dataset size according to formulas from the original papers (:math:`\max(250, 1.25\% \times k(n-k))` in CLARANS and :math:`\max(1, \lfloor 250/k \rfloor, 2.5\% \times (n-k))` in FastCLARANS). An explicit integer value (such as ``max_neighbors=200``) is mainly useful when you need a fixed ceiling on search steps for very large inputs.

Search mechanics
----------------

The k-medoids search space can be viewed as an undirected graph :math:`G_{n,k} = (V, E)`:

* Vertices: Each vertex :math:`v \in V` represents a set of :math:`k` medoids from :math:`n` samples. The total number of vertices is :math:`\binom{n}{k}`.
* Edges: An edge connects two vertices if their medoid sets differ by exactly one point (one swap :math:`m_j \leftrightarrow x_c`). Each vertex has :math:`k(n-k)` neighbors.
* Exhaustive search: Standard PAM checks all :math:`k(n-k)` neighbors at each step to find the steepest descent, which costs :math:`O(k(n-k)^2)` per iteration.
* CLARANS search: CLARANS checks randomly sampled neighbors. As soon as a neighbor reduces total distance cost, it makes the swap immediately (first-choice hill climbing). If ``max_neighbors`` consecutive samples yield no improvement, the search stops at a local minimum.
* FastCLARANS search: FastCLARANS samples candidate replacement points and evaluates the swap delta against all :math:`k` current medoids simultaneously using FastPAM1 equations. A single :math:`O(n)` pass over the data evaluates :math:`k` neighbors at once.
