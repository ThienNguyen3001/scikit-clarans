User Guide
==========

This guide covers the basics of using **scikit-clarans** for your clustering tasks.

What is CLARANS?
----------------

**CLARANS** stands for *Clustering Large Applications based on RANdomized Search*. 

Think of it as a middle-ground between:

*   **PAM (Partitioning Around Medoids):** High quality, but slow on large data.
*   **CLARA (Clustering Large Applications):** Faster on large data, but works on fixed samples, potentially missing better clusterings.

CLARANS explores the graph of possible solutions randomly. It doesn't check *every* neighbor of a node (a set of medoids), but only a random subset. This makes it scalable while better avoiding local minima approaches.

Quick Start
-----------

Here is a complete example to get you clustering in seconds.

.. code-block:: python

    from clarans import CLARANS
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # 1. Prepare your data
    # We'll generate 500 samples with 4 distinct centers
    X, _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)

    # 2. Initialize CLARANS
    # We want to find 4 clusters. 
    model = CLARANS(
        n_clusters=4, 
        numlocal=3, 
        init='k-medoids++', 
        random_state=42
    )

    # 3. Fit the model
    model.fit(X)

    # 4. Analyze results
    print(f"Medoid Indices: {model.medoid_indices_}")
    print(f"Labels: {model.labels_[:10]}...")

Configuration
-------------

The ``CLARANS`` class offers several parameters to tune performance vs. quality:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``n_clusters``
     - The number of clusters (medoids) to find.
   * - ``numlocal``
     - Number of local optima to search for. Higher usually means better quality but slower execution.
   * - ``maxneighbor``
     - Max neighbors to check per node. Defaults to a percentage of dataset size if not set.
   * - ``init``
     - Introduction strategy (e.g., ``'k-medoids++'``, ``'build'``, ``'random'``).

Tips for Best Results
---------------------

*   **Initialization matters:** Using ``init='k-medoids++'`` or ``'build'`` often converges faster to better solutions than pure random.
*   **Tuning parameters:** If your results vary too much between runs, try increasing ``numlocal`` to explore more local minima.

FastCLARANS
-----------

**FastCLARANS** is a faster variant based on Schubert & Rousseeuw (2021). It provides
significant speedups by using the FastPAM1 optimization strategy.

.. code-block:: python

    from clarans import FastCLARANS

    model = FastCLARANS(n_clusters=4, numlocal=3, random_state=42)
    model.fit(X)

**Key improvements over CLARANS:**

*   **Smarter sampling:** Instead of sampling random (medoid, non-medoid) pairs, FastCLARANS samples only non-medoid candidates and evaluates swaps with all k medoids at once.
*   **O(k) speedup:** Each candidate evaluation explores k edges of the search graph in the time CLARANS explores one.
*   **Memory efficient:** Computes distances on-the-fly (O(n) memory) rather than precomputing a full distance matrix (O(n²)).
*   **Better quality:** By exploring more of the search space per iteration, FastCLARANS often finds better solutions.

**When to use FastCLARANS vs CLARANS:**

*   Use **FastCLARANS** when you have low-dimensional data with cheap distance metrics (e.g., Euclidean).
*   Use **CLARANS** when distance computation is very expensive or when you need maximum memory efficiency.

For more hands-on recipes and runnable examples (including a Jupyter notebook with interactive demos), see :doc:`examples`.
