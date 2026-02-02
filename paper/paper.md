---
title: 'scikit-clarans: A Python Library for CLARANS Clustering'
tags:
  - Python
  - clustering
  - machine learning
  - data mining
  - k-medoids
  - CLARANS
authors:
  - name: Ngoc Thien Nguyen
    orcid: 0009-0003-3548-6303
    affiliation: 1
    corresponding: true 
affiliations:
 - name: University of Economics Ho Chi Minh City, Vietnam
   index: 1
date: 02 February 2026
bibliography: paper.bib
---

# Summary

Clustering is a fundamental technique in data mining and unsupervised machine learning, used to group similar objects into sets. Among clustering methods, partitioning algorithms like $k$-means and $k$-medoids are widely used. While $k$-means is efficient, it is sensitive to outliers and limited to Euclidean distances. $k$-medoids algorithms, such as PAM (Partitioning Around Medoids), are more robust and support arbitrary distance metrics but effectively scale poorly ($O(k(N-k)^2)$) to large datasets [@kaufman1990finding].

`scikit-clarans` is a Python library that implements the **CLARANS** (Clustering Large Applications based on RANdomized Search) algorithm [@ng2002clarans]. CLARANS acts as a bridge between the high solution quality of PAM and the efficiency required for larger datasets. It views the process of finding $k$ medoids as searching through a graph where each node is a potential set of medoids. By detecting local minima through randomized search rather than exhaustive enumeration, CLARANS achieves a balance between computational speed and clustering quality.

Designed to be fully compatible with the `scikit-learn` ecosystem [@scikit-learn], `scikit-clarans` allows users to easily integrate this powerful clustering algorithm into standard machine learning pipelines, leveraging the familiar `fit`, `predict`, and `transform` API.

# Statement of need

As datasets grow in size and complexity, the need for robust clustering algorithms that can handle non-Euclidean distances and outliers becomes critical. While $k$-means is fast, it is often unsuitable for domains requiring specific dissimilarity measures (e.g., categorical data, biological sequences). Traditional $k$-medoids implementations like PAM provide high quality but become computationally prohibitive as $N$ increases.

`scikit-clarans` targets data scientists, researchers, and students who require:

1.  **Robustness**: A clustering algorithm less sensitive to outliers than $k$-means.
2.  **Flexibility**: The ability to use arbitrary distance metrics (Manhattan, Cosine, or precomputed distance matrices).
3.  **Scalability**: A method that scales better than PAM for medium-to-large datasets.
4.  **Integration**: A tool that fits seamlessly into the `scikit-learn` workflow (Pipelines, GridSearchCV) without the need for custom wrappers.

# State of the field

In the Python ecosystem, `scikit-learn` [@scikit-learn] is the standard for machine learning. However, its native support for $k$-medoids is limited. 

- **`scikit-learn-extra`** provides a `KMedoids` implementation (typically PAM-based), which offers exact solutions but suffers from the aforementioned performance bottlenecks on large data.

- **`pyclustering`** [@novikov2019pyclustering] offers a wide range of clustering algorithms, including CLARANS. While powerful, its API differs significantly from `scikit-learn`, making integration into existing workflows (e.g., cross-validation, pipelines) less straightforward.

`scikit-clarans` fills this niche by providing a dedicated, `scikit-learn`-native implementation of CLARANS. It strictly adheres to `scikit-learn`'s API design guidelines (regarding state management, parameter validation, and input handling), ensuring it can be used as a drop-in replacement for `KMeans` or other clusterers.

# Software Design

The library is designed with modularity and usability in mind:

1.  **API Consistency**: The main classes, `CLARANS` and `FastCLARANS`, inherit from `sklearn.base.ClusterMixin`, `TransformerMixin`, and `BaseEstimator`. This ensures methods like `fit()`, `fit_predict()`, and `transform()` behave exactly as users expect.
2.  **Algorithm Variants**:
    *   **`CLARANS`**: The standard implementation suitable for datasets where memory is a constraint or distance calculation is cheap. It computes distances on-the-fly or uses a precomputed matrix.
    *   **`FastCLARANS`**: An optimized variant that precomputes and caches the distance matrix. This offers significant speedups for datasets that fit in memory by utilizing vectorized NumPy operations [@harris2020array] for delta-cost calculations, similar to optimizations proposed in modern $k$-medoids research [@schubert2019faster].
3.  **Initialization Strategies**: The library supports multiple initialization methods to avoid poor local optima, including random selection, `k-medoids++` (analogous to `k-means++` [@arthur2007kmeans] for better initial spread), and heuristic based initialization.
4.  **Distance Metric Support**: Leverages `sklearn.metrics.pairwise_distances` (powered by SciPy [@virtanen2020scipy]) to support a wide array of metrics (Euclidean, Manhattan, Cosine, etc.) and sparse matrix inputs.

# Research Impact Statement

`scikit-clarans` significant contributions to the research community and data science practitioners:

1.  **Benchmarking & Reproducibility**: By providing a standard-compliant, open-source implementation of CLARANS, the library facilitates fair and reproducible benchmarking against modern clustering algorithms. Researchers can now easily compare their new methods against CLARANS within the same software framework.
2.  **Educational Tool**: The codebase serves as a clear reference implementation for students and educators teaching randomized algorithms and clustering techniques. Its integration with `scikit-learn` makes it accessible for classroom assignments without complex setup.
3.  **Bridge for Domain Scientists**: Scientists in fields like bioinformatics and spatial data mining, where data often comes with custom non-Euclidean distance matrices, can utilize the efficient graph-search capabilities of CLARANS without needing deep algorithmic expertise.
4.  **Optimization Research**: The inclusion of `FastCLARANS` demonstrates the application of modern caching and vectorization techniques to classical algorithms, providing a case study in optimizing Python-based scientific software.

# Methodology

CLARANS interprets the clustering problem as a search in a graph $G_{n,k}$.

- **Nodes**: Each node represents a set of $k$ medoids.
- **Edges**: Two nodes are connected if their sets of medoids differ by exactly one object.

Unlike PAM, which evaluates all neighbors of the current node (swapping every medoid with every non-medoid), CLARANS draws a sample of neighbors controlled by the `maxneighbors` parameter. If a neighbor with a lower cost (sum of distances from objects to their nearest medoid) is found, the algorithm moves to that neighbor. This process repeats until a local minimum is found. To avoid getting stuck in poor local optima, the search is repeated `numlocal` times starting from different random nodes.

The algorithm parameters are:

- `n_clusters` ($k$): Number of clusters.
- `numlocal`: Number of local searches.
- `maxneighbors`: Maximum number of neighbors examined.

# Example Usage

The following example demonstrates how to use `CLARANS` to cluster a synthetic dataset and extract the medoids.

```python
from clarans import CLARANS
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 1. Generate synthetic data
X, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=42)

# 2. Initialize the model
# Using 'k-medoids++' initialization for better convergence
clarans = CLARANS(n_clusters=5, numlocal=3, init='k-medoids++', random_state=42)

# 3. Fit the model
clarans.fit(X)

# 4. Access results
print(f"Medoid Indices: {clarans.medoid_indices_}")
print(f"Labels: {clarans.labels_[:10]}...")

# 5. Evaluate
score = silhouette_score(X, clarans.labels_)
print(f"Silhouette Score: {score:.3f}") # Based on work by @rousseeuw1987silhouettes

# 6. Usage in a Pipeline (optional)
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# pipe = Pipeline([('scaler', StandardScaler()), ('clarans', clarans)])
# pipe.fit(X)
```

# AI usage disclosure

We acknowledge the use of generative AI tools (specifically GitHub Copilot) in the development of this software and paper. These tools were used for:

1.  **Code Assistance**: Generating boilerplate code, unit tests, and type hints.
2.  **Documentation**: Assisting in drafting docstrings and refactoring documentation for clarity.

All AI-generated content was manually reviewed, verified, and refined by the authors to ensure accuracy and scientific validity. The core logic of the CLARANS algorithm and the software architecture decisions are the work of the authors.

# Acknowledgements

We acknowledge the foundational work by Ng and Han [@ng2002clarans] in developing the CLARANS algorithm. We also thank the `scikit-learn` community for providing the robust framework upon which this library is built.
