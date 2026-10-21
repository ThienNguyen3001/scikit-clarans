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
   * - ``n_jobs``
     - Both
     - ``None``
     - Number of parallel jobs to run across ``num_local`` restarts during ``fit``. ``None`` means 1 unless in a ``joblib.parallel_backend`` context; ``-1`` uses all available CPU threads.

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
  * Array-like: User-provided array of initial medoid coordinates of shape ``(n_clusters, n_features)``, or 1D integer indices of shape ``(n_clusters,)`` when ``metric='precomputed'``.

    * **Coordinate Snapping**: Because medoids must be actual observations from the training dataset :math:`X`, user-provided coordinates are automatically snapped to their nearest unique training sample using the configured distance metric.
    * **Duplicate Back-filling**: If multiple coordinates snap to the same training sample, ``scikit-clarans`` issues a ``UserWarning`` and automatically back-fills the missing slots with randomly chosen available training points, guaranteeing exactly :math:`k` distinct medoids without breaking downstream pipelines.

Number of restarts (``num_local``):
  For randomized initialization (``'k-medoids++'`` or ``'random'``), increasing ``num_local`` to 3 to 5 helps find better local minima when solutions vary across runs.
  
  .. note::
     Deterministic initialization methods such as ``'build'``, ``'heuristic'``, or explicit coordinate arrays always produce the same starting points. ``scikit-clarans`` pre-computes and caches these initial medoids once before the search loop to avoid redundant :math:`O(n^2)` recalculations across restarts. Set ``num_local=1`` to save computation.

Neighbor sampling (``max_neighbors``):
  The default ``max_neighbors='auto'`` scales with the dataset size according to formulas from the original papers (:math:`\max(250, 1.25\% \times k(n-k))` in CLARANS and :math:`\max(1, \lfloor 250/k \rfloor, 2.5\% \times (n-k))` in FastCLARANS). An explicit integer value (such as ``max_neighbors=200``) is mainly useful when you need a fixed ceiling on search steps for very large inputs.

Parallel execution (``n_jobs``):
  When ``num_local > 1``, setting ``n_jobs`` (e.g., ``n_jobs=-1`` or ``n_jobs=4``) parallelizes independent local search restarts across worker threads using ``joblib.Parallel(prefer="threads")``.

  * **GIL release & speedup**: Because the core swap evaluation loops run in Cython with ``nogil`` blocks, multithreaded restarts achieve true multicore speedups without inter-process communication (IPC) or data serialization overhead.
  * **Bit-for-bit reproducibility**: Seeds for each local search are partitioned deterministically from ``random_state`` before parallel dispatch, ensuring identical clustering results regardless of the ``n_jobs`` setting (e.g., ``n_jobs=1`` and ``n_jobs=-1`` yield bit-for-bit identical outputs).
  * **Zero-overhead sequential fast path**: When ``num_local=1`` or the effective job count is 1, execution avoids worker thread dispatch overhead completely.

Distance metrics and routing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``scikit-clarans`` automatically routes distance computations based on the input data type, matrix sparsity, and chosen metric:

1. **Dense array metrics**:
   Standard distance functions (``'euclidean'``, ``'manhattan'``, ``'cosine'``, ``'chebyshev'``, ``'minkowski'``, ``'mahalanobis'``) are routed to SciPy's optimized ``cdist``. Metric aliases are mapped automatically (such as ``'cityblock'`` and ``'l1'`` to ``'manhattan'``, or ``'l2'`` to ``'euclidean'``). Keyword parameters such as ``{'p': 3}`` or ``{'VI': VI}`` are passed via ``metric_params``.

2. **Sparse matrix inputs (scipy.sparse)**:
   When ``X`` is a CSR or CSC sparse matrix, distance evaluations are routed to scikit-learn's ``DistanceMetric`` without densifying the full matrix, maintaining an :math:`O(n)` memory footprint.

3. **Missing data (NaN) handling**:
   When inputs contain missing values (NaN), configure ``metric='nan_euclidean'``. Any dimension where either sample is missing is excluded from the distance sum, and the Euclidean distance is scaled by :math:`\sqrt{d / d_{\text{valid}}}`.

4. **Custom callable functions**:
   Supply any Python distance function ``metric=func`` accepting two 1D vectors ``(u, v) -> float``. For example, a custom Haversine function can cluster geographical latitude and longitude coordinates on the Earth's surface.

5. **Precomputed distance matrices**:
   Pass a pairwise distance matrix with ``metric='precomputed'``. The algorithm validates symmetry using fast Cython kernels (``is_matrix_symmetric``) and handles both symmetric and directed/asymmetric matrices (:math:`D \neq D^T`). When performing inference with ``predict()`` or ``transform()`` on new query points, you can provide either the full matrix of shape :math:`(n_{\text{test}}, n_{\text{train}})` or a compact submatrix of shape :math:`(n_{\text{test}}, k)` matching only the :math:`k` medoids.

Fitted attributes
-----------------

After calling ``fit(X)``, both ``CLARANS`` and ``FastCLARANS`` store the clustering results and search statistics in estimator attributes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``cluster_centers_``
     - {ndarray, sparse matrix}
     - Coordinates of the selected medoids with shape ``(n_clusters, n_features)``. Defined only when ``metric != 'precomputed'``.
   * - ``medoid_indices_``
     - ndarray
     - Row indices of the selected medoids in the training data ``X`` with shape ``(n_clusters,)``.
   * - ``labels_``
     - ndarray
     - Cluster labels assigned to each sample in the training set with shape ``(n_samples,)``.
   * - ``inertia_``
     - float
     - Sum of distances from each point to its assigned medoid according to the chosen metric.
   * - ``max_neighbors_``
     - int
     - Effective threshold of non-improving neighbor evaluations per restart. Equals the resolved integer value when ``max_neighbors='auto'``.
   * - ``neighborhood_size_``
     - int
     - Search neighborhood size: :math:`k(n-k)` in ``CLARANS`` (graph transitions) or :math:`n-k` in ``FastCLARANS`` (candidate points).
   * - ``n_iter_``
     - int
     - Number of candidate evaluations tested during the best local search restart.
   * - ``n_swaps_``
     - int
     - Number of accepted medoid swaps performed during the best local search restart.
   * - ``n_features_in_``
     - int
     - Number of input features seen during ``fit``. When ``metric='precomputed'``, equals the number of training samples (column dimension).

Attribute details and conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inertia calculation:
  ``inertia_`` represents the sum of unsquared distances :math:`\sum_{i=1}^n \min_{m \in M} d(x_i, m)` using the chosen distance metric. Standard ``KMeans`` in scikit-learn computes the sum of squared Euclidean distances (:math:`\sum d^2`). When comparing inertia across estimators or distance functions, account for the difference between linear and squared scale.

Coordinates with precomputed metrics:
  When ``metric='precomputed'``, the input ``X`` is a distance matrix of shape ``(n_samples, n_samples)``. Because the original feature coordinates are not passed to ``fit``, ``cluster_centers_`` is not defined (hasattr returns ``False``). Use ``medoid_indices_`` to retrieve the row positions of the selected medoids in your original data.

Precomputed inference and best practices:
  When serving models with ``metric='precomputed'``, ``predict()`` and ``transform()`` accept two matrix shapes:

  * **Full matrix** of shape ``(n_test, n_train)``: Distance matrix from query samples to all training samples.
  * **Compact matrix** of shape ``(n_test, k)``: Distance matrix from query samples directly to the ``k`` fitted medoids. This reduces distance evaluations and memory consumption by a factor of :math:`n_{\text{train}} / k` (typically 10x to 1000x faster).

  .. warning::
     **Important considerations when using precomputed inference:**

     1. **Medoid preservation**: Because the model only receives distance matrices during ``fit()``, it does not store the original raw objects. You must retain your original training items and extract the medoids using ``model.medoid_indices_`` in order to compute query distances later.
     2. **Column alignment**: When passing a compact matrix of shape ``(n_test, k)``, column ``j`` must strictly correspond to medoid ``model.medoid_indices_[j]``. A mismatched column order will cause silent misclassifications. Using a ``pandas.DataFrame`` with matching column names is recommended to guard against accidental column reordering.
     3. **Alternative for non-vector data**: If your dataset consists of complex domain objects (e.g., DNA sequences, graphs, raw strings), consider passing a custom distance callable (``metric=func``) directly to the estimator instead of precomputing distance matrices. When using a callable metric, ``cluster_centers_`` automatically stores the raw medoid objects, enabling standard ``predict(X_test)`` where distances are computed internally without manual matrix management.

Neighbor attributes and auto-scaling:
  The constructor parameter ``max_neighbors`` accepts an integer or the string ``'auto'``. The fitted attribute ``max_neighbors_`` (with a trailing underscore) always stores the concrete integer limit used during the search.

  The total neighborhood size ``neighborhood_size_`` differs between estimators because of their search formulation:

  * In ``CLARANS``, each search step samples a pair of (medoid, non-medoid), so ``neighborhood_size_`` equals the number of edges in graph :math:`G_{n,k}`, namely :math:`k(n-k)`.
  * In ``FastCLARANS``, each step samples a non-medoid candidate point and evaluates all :math:`k` medoid swaps simultaneously, so ``neighborhood_size_`` equals :math:`n-k`.

Iteration and swap counters:
  In accordance with scikit-learn clustering conventions (e.g., ``KMeans``, ``KMedoids``), ``n_iter_`` and ``n_swaps_`` record the candidate evaluations and accepted medoid swaps performed during the best local search restart (the run that achieved minimum ``inertia_``).

Monitoring search progress
--------------------------

Both ``CLARANS`` and ``FastCLARANS`` provide console logging through the ``verbose`` parameter to monitor local searches during ``fit``.

Summary table (verbose=1)
^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``verbose=1`` (or ``verbose=True``) prints a configuration header, a summary table with one row per local search, and a final line marking the best restart:

.. code-block:: text

    [CLARANS] n=1000, k=4, metric=euclidean, num_local=3, max_neighbors=250/3984 (6.3%)
        #            Cost  Swaps  Evals   Time(s)
        1       150.12345*     4     80     0.012s  converged
        2       148.54321*     5    110     0.015s  converged
        3       152.00000      2     65     0.009s  converged
      Best: #2 | Totals: 11 swaps, 255 evals, 0.038s

The header reports the dataset size (:math:`n`), number of clusters (:math:`k`), distance metric, total restarts (``num_local``), and the sampled neighbor budget (``max_neighbors``) relative to the total neighborhood size.

Table columns:

* ``#``: 1-based index of the restart.
* ``Cost``: Final clustering inertia (sum of distances) for that restart. An asterisk (``*``) flags an iteration that set a new lowest cost across all completed restarts.
* ``Swaps``: Number of accepted medoid swaps.
* ``Evals``: Total candidate evaluations tested during the restart.
* ``Time(s)``: Wall-clock duration of the restart.
* Stopping status:

  * ``converged``: The search evaluated ``max_neighbors`` consecutive non-improving candidates without finding a cost reduction.
  * ``exhausted``: The search tested every available non-medoid sample before reaching ``max_neighbors``.

The summary footer displays the restart index that achieved the lowest cost, followed by cumulative totals for accepted swaps, evaluations, and overall execution time.

Step-by-step traces (verbose=2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``verbose=2`` prints each accepted swap in real time:

.. code-block:: text

    [CLARANS] n=1000, k=4, metric=euclidean, num_local=2, max_neighbors=250/3984 (6.3%)
      Restart 1/2 (init cost: 234.56789):
          swap   1 | eval     8 | cost      210.12345 | diff    -24.44444
          swap   2 | eval    25 | cost      180.50000 | diff    -29.62345
         1       180.50000*  swaps=2, evals=275, 0.021s (converged)
      Restart 2/2 (init cost: 245.11111):
          swap   1 | eval    12 | cost      195.43210 | diff    -49.67901
         2       195.43210   swaps=1, evals=262, 0.019s (converged)
      Best: #1 | Totals: 3 swaps, 537 evals, 0.041s

In addition to the restart summary, this level records:

* ``init cost``: Starting inertia computed right after medoid initialization.
* ``swap``: Sequence number of the accepted swap within the current restart.
* ``eval``: Running count of neighbor evaluations examined when the improving swap was found.
* ``cost``: New total distance cost after making the swap.
* ``diff``: Cost change (:math:`\Delta < 0`) produced by the swap.

Search mechanics
----------------

The k-medoids search space can be viewed as an undirected graph :math:`G_{n,k} = (V, E)`:

* Vertices: Each vertex :math:`v \in V` represents a set of :math:`k` medoids from :math:`n` samples. The total number of vertices is :math:`\binom{n}{k}`.
* Edges: An edge connects two vertices if their medoid sets differ by exactly one point (one swap :math:`m_j \leftrightarrow x_c`). Each vertex has :math:`k(n-k)` neighbors.
* Exhaustive search: Standard PAM checks all :math:`k(n-k)` neighbors at each step to find the steepest descent, which costs :math:`O(k(n-k)^2)` per iteration.
* CLARANS search: CLARANS checks randomly sampled neighbors. As soon as a neighbor reduces total distance cost, it makes the swap immediately (first-choice hill climbing). If ``max_neighbors`` consecutive samples yield no improvement, the search stops at a local minimum.
* FastCLARANS search: FastCLARANS samples candidate replacement points and evaluates the swap delta against all :math:`k` current medoids simultaneously using FastPAM1 equations. A single :math:`O(n)` pass over the data evaluates :math:`k` neighbors at once.
