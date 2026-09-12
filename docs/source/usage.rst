User Guide
==========

This guide covers the essentials of clustering with **scikit-clarans**, explaining practical use cases, algorithm selection, parameter tuning, and the core search mechanics.

Why k-Medoids? (k-Medoids vs. k-Means)
--------------------------------------

While **k-Means** is widely used, it has fundamental limitations in practical machine learning workflows:

1. **Artificial Centroids vs. Real Data Points**:
   k-Means averages points in Euclidean space to produce cluster "centroids" that do not exist in the original data. In contrast, **k-medoids** restricts cluster centers (medoids) to **actual data points from your dataset**. This provides immediate interpretability:
   
   * In customer analytics: The medoid is a real customer profile.
   * In text clustering: The medoid is an actual representative document.
   * In bioinformatics: The medoid is a real sequenced molecule or patient case.

2. **Outlier Robustness**:
   k-Means minimizes the sum of squared Euclidean distances (:math:`\sum \|x_i - \mu\|^2`). Squaring distances gives extreme outliers disproportionate leverage, dragging centroids away from true cluster bodies. k-Medoids minimizes the sum of absolute pairwise distances:

   .. math::

      \min_{M \subset X, |M|=k} \sum_{i=1}^n \min_{m \in M} d(x_i, m)

   This gives k-medoids a substantially higher breakdown point against anomalies and noise.

3. **Arbitrary Distance Metrics**:
   k-Means is geometrically tied to Euclidean distance. k-Medoids works seamlessly with **any valid distance metric** (e.g., Cosine similarity for embeddings, Manhattan distance for grid layouts, or precomputed distance matrices).

.. note::

   **A note on** ``inertia_``: In ``scikit-learn``'s ``KMeans``, ``inertia_`` represents the sum of *squared* Euclidean distances. In ``scikit-clarans``, ``inertia_`` represents the sum of *unsquared* distances from each sample to its assigned medoid according to the chosen metric.

Quick Start
-----------

Here is a quick example demonstrating clustering with ``CLARANS``:

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

Choosing an Estimator: CLARANS vs. FastCLARANS
----------------------------------------------

``scikit-clarans`` provides two main estimators:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - ``CLARANS``
     - ``FastCLARANS`` (Recommended)
   * - **Original Paper**
     - Ng & Han (2002)
     - Schubert & Rousseeuw (2021)
   * - **Sampling Strategy**
     - Samples random (medoid, non-medoid) pairs
     - Samples non-medoids; tests all :math:`k` medoids at once
   * - **Delta Cost Update**
     - Single swap evaluation (:math:`O(n)` with ``cache=True``, :math:`O(n \cdot k)` with ``cache=False``)
     - FastPAM1 vectorized delta cost across all :math:`k` medoids
   * - **Memory Footprint**
     - :math:`O(n)` on-the-fly
     - :math:`O(n)` on-the-fly
   * - **Best For**
     - Baseline / reproduction of Ng & Han (2002)
     - Recommended choice for small-to-medium datasets and research experimentation

.. note::
   **Dataset Size & Scalability Limitation**

   While ``FastCLARANS`` substantially outperforms classic ``CLARANS`` by testing all :math:`k` medoid swaps at once with :math:`O(n)` memory, the current library is implemented in pure Python and NumPy. It is intended for **educational exploration, experimentation, and small-to-medium datasets** (up to tens of thousands of samples). It is **not yet optimized for large-scale Big Data pipelines** (such as :math:`N \gg 10^5`).

Quick example with ``FastCLARANS``:

.. code-block:: python

    from clarans import FastCLARANS

    # FastCLARANS evaluates k graph edges per candidate evaluation
    fast_model = FastCLARANS(n_clusters=4, num_local=3, random_state=42)
    fast_model.fit(X)

Configuration & Hyperparameter Tuning
-------------------------------------

Both estimators share core hyperparameters to balance execution speed and clustering quality, with ``CLARANS`` providing an additional ``cache`` parameter:

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
     - Maximum non-improving neighbors to check per search. Defaults to :math:`\max(250, 1.25\% \times k(n-k))` in CLARANS and :math:`\max(250, 2.5\% \times (n-k))` in FastCLARANS.
   * - ``init``
     - Both
     - ``k-medoids++``
     - Initialization strategy (``k-medoids++``, ``random``, ``build``, ``heuristic``, or array-like).
   * - ``metric``
     - Both
     - ``euclidean``
     - Distance metric to use (e.g., ``euclidean``, ``manhattan``, ``cosine``).
   * - ``cache``
     - CLARANS only
     - ``True``
     - Whether to use distance caching (:math:`d_1, d_2`) for :math:`O(n)` swap evaluations. If ``False``, recalculates total cost from scratch via ``calculate_cost()`` in :math:`O(n \cdot k)`.

Practical Tuning Tips
^^^^^^^^^^^^^^^^^^^^^^

* **Distance Caching** (``cache`` in CLARANS):
  ``CLARANS`` supports an optional ``cache`` parameter (default ``True``). Note that ``FastCLARANS`` does not expose a ``cache`` parameter because its vectorized FastPAM1 delta calculation inherently tracks :math:`d_1` and :math:`d_2`:
  
  * ``cache=True`` *(Recommended)*: Maintains an on-the-fly cache of nearest (:math:`d_1`) and second-nearest (:math:`d_2`) medoid distances for all samples. Each candidate swap is evaluated in :math:`O(n \cdot d)` operations without recomputing distances to unchanged medoids. This yields a 1.5x–2.5x speedup with identical mathematical clustering results while maintaining a lean :math:`O(n)` memory footprint.
  * ``cache=False``: Evaluates each candidate swap by recalculating the total clustering cost from scratch using ``calculate_cost`` in :math:`O(n \cdot k \cdot d)`. This reproduces the exact classic baseline behavior of Ng & Han (2002).

* **Initialization Strategy** (``init``):
  
  * ``k-medoids++`` *(Default, Recommended)*: Probabilistic seeding proportional to squared distance. Fast and memory-friendly (:math:`O(n \cdot k)`).
  * ``random``: Pure uniform sampling. Very fast, but typically requires increasing ``num_local`` to achieve comparable clustering quality.
  * ``build``: Classic PAM greedy seeding. Excellent solution quality on small datasets, but computes the full pairwise distance matrix (:math:`O(n^2)` time and memory). Avoid on large datasets (:math:`n > 5000`).
  * ``heuristic``: Selects the :math:`k` most central data points with the smallest total distance to all others (:math:`O(n^2)` time and memory). Avoid on large datasets (:math:`n > 5000`).
  * ``array-like``: Pass custom coordinates of shape ``(n_clusters, n_features)`` or pre-defined medoid indices to inject prior domain knowledge.

* **Number of Restarts** (``num_local``):
  If cluster assignments fluctuate between runs or the objective value (``inertia_``) is inconsistent, increase ``num_local`` to 3–5 when using randomized initialization (``k-medoids++`` or ``random``).
  
  .. note::
     **Deterministic initialization**: Strategies such as ``build``, ``heuristic``, or explicit centroid arrays are deterministic. Setting ``num_local > 1`` with these strategies will start all local searches from the exact same medoids. It is recommended to use ``num_local=1`` with deterministic initialization to conserve compute resources.

* **Candidate Exploration** (``max_neighbors``):
  Leaving ``max_neighbors='auto'`` (the default) is strongly recommended for almost all use cases. It automatically adapts to the problem geometry based on empirical ratios from the original papers (:math:`1.25\% \times k(n-k)` in CLARANS and :math:`2.5\% \times (n-k)` in FastCLARANS). You only need to explicitly specify an integer for ``max_neighbors`` (e.g., ``max_neighbors=200``) if you must enforce a hard upper bound on execution time on very large datasets.

How It Works (Under the Hood)
-----------------------------

Understanding the search graph :math:`G_{n,k}` helps developers reason about convergence:

1. **The Search Graph**:
   The problem space is modeled as an undirected graph :math:`G_{n,k} = (V, E)`:

   * Each vertex :math:`v \in V` represents a candidate set of :math:`k` medoids (:math:`|V| = \binom{n}{k}`).
   * Two vertices are connected by an edge if their medoid sets differ by exactly one point (a single swap :math:`(m_j \leftrightarrow x_c)`).
   * Each node has exactly :math:`k(n-k)` neighbors.

2. **Why Exhaustive PAM is Slow**:
   Standard PAM examines all :math:`k(n-k)` neighbors at each step to find the steepest descent, costing :math:`O(k(n-k)^2)` per iteration. This becomes intractable for large datasets.

3. **How CLARANS Accelerates Search**:
   Instead of checking all :math:`k(n-k)` neighbors, CLARANS draws random candidate neighbors. As soon as it finds a neighbor that reduces the clustering cost, it immediately transitions to that node (first-choice hill climbing). If :math:`\text{max\_neighbors}` consecutive random neighbors fail to improve the cost, the search terminates at a local optimum. The process repeats :math:`\text{num\_local}` times from new random starts.

4. **How FastCLARANS Improves Exploration**:
   FastCLARANS utilizes the FastPAM1 formulation: by tracking the nearest and second-nearest medoids for each sample, it computes the swap delta for **all** :math:`k` **medoids simultaneously** in a single :math:`O(n)` pass over the data. This evaluates :math:`k` graph edges in the time CLARANS evaluates one.

Next Steps
----------

* Check out the runnable recipes in :doc:`examples` (including Colab notebooks).
* Review the full parameter specifications in :doc:`api`.
