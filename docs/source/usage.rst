User Guide
==========

This guide provides a quick introduction to using ``scikit-clarans`` for clustering.

CLARANS Algorithm
-----------------

CLARANS (Clustering Large Applications based on RANdomized Search) is a partitioning method that searches for $k$ medoids. It is an extension of the PAM (Partitioning Around Medoids) algorithm but uses randomized search to handle larger datasets more efficiently.

Basic Usage
-----------

You can use ``CLARANS`` just like any other scikit-learn clusterer (e.g., ``KMeans``).

.. code-block:: python

    from clarans import CLARANS
    from sklearn.datasets import make_blobs

    # 1. Generate sample data
    X, y = make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)

    # 2. Initialize the model
    # n_clusters: Number of clusters (k)
    # numlocal: Number of local minima to find
    # maxneighbor: Max neighbors to examine (randomly)
    clarans = CLARANS(n_clusters=4, numlocal=3, maxneighbor=None, init='k-medoids++', random_state=42)

    # 3. Fit to data
    clarans.fit(X)

    # 4. Get results
    print("Cluster Centers (Medoids):", clarans.cluster_centers_)
    print("Labels:", clarans.labels_)
    print("Medoid Indices in X:", clarans.medoid_indices_)


Key Parameters
--------------

- **n_clusters**: The number of clusters to form.
- **numlocal**: The number of local searches to perform. Higher values increase the chance of finding a global minimum but take more time.
- **maxneighbor**: The maximum number of neighbors to examine during each local search. If ``None``, it defaults to a heuristic based on dataset size.
- **init**: Initialization method. Options include ``'random'``, ``'heuristic'``, ``'k-medoids++'``, and ``'build'``.

Choosing Parameters
-------------------

- For large datasets, increase ``numlocal`` and keep ``maxneighbor`` reasonable to balance speed and quality.
- ``'k-medoids++'`` initialization often leads to faster convergence and better initial configurations.
