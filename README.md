# scikit-clarans

> A scikit-learn compatible implementation of the CLARANS (Clustering Large Applications based on RANdomized Search) algorithm.

CLARANS is a partitioning clustering algorithm that extends the k-medoids approach (like PAM) to handle large datasets effectively. It views the process of finding $k$ medoids as searching through a graph where each node is a potential set of $k$ medoids. By randomized search, CLARANS avoids the computational cost of checking every neighbor as PAM does, while avoiding the local minima issues of CLARA.

## Key Features

- **Scikit-learn Compatible**: Inherits from `BaseEstimator` and `ClusterMixin`, fully passing `check_estimator` tests.
- **Flexible Initialization**: Supports multiple initialization strategies including `random`, `heuristic`, `k-medoids++`, and `build` (Greedy).
- **Custom Metrics**: Supports all distance metrics available in `sklearn.metrics.pairwise_distances` (e.g., euclidean, manhattan, cosine).
- **Reproducibility**: Fully deterministic when `random_state` is provided.

## Installation

### From Source

Ensure you have `numpy` and `scikit-learn` installed. You can install this package directly from the source:

```bash
git clone https://github.com/ThienNguyen3001/scikit-clarans.git
cd scikit-clarans
pip install .
```

For development (editable mode):

```bash
pip install -e .
```

## Quick Start

```python
from clarans import CLARANS
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Generate sample data
X, y = make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)

# 2. Initialize CLARANS
# Uses 'k-medoids++' for smarter initialization
clarans = CLARANS(n_clusters=4, numlocal=3, maxneighbor=None, init='k-medoids++', random_state=42)

# 3. Fit the model
clarans.fit(X)

# 4. Access results
print("Cluster Medoid Indices:", clarans.medoid_indices_)
print("Cluster Centers (Coordinates):", clarans.cluster_centers_)
print("Labels:", clarans.labels_[:10])

# 5. Predict new points
new_points = [[0, 0], [10, 10]]
predictions = clarans.predict(new_points)
print("Predictions:", predictions)
```

## API Reference

### `CLARANS(n_clusters=8, numlocal=2, maxneighbor=None, max_iter=300, init='random', metric='euclidean', random_state=None)`

#### Parameters

- **`n_clusters`** *(int, default=8)*: The number of clusters to form as well as the number of medoids to generate.
- **`numlocal`** *(int, default=2)*: The number of local minima to find. The algorithm runs the local search `numlocal` times and returns the best result.
- **`maxneighbor`** *(int, default=None)*: Maximum number of neighbors to examine during local search. If `None`, it is set to `max(250, 1.25% * k*(n-k))`.
- **`max_iter`** *(int, default=300)*: Maximum number of hops (swaps) allowed in a single local search to prevent infinite loops.
- **`init`** *({'random', 'heuristic', 'k-medoids++', 'build', array-like}, default='random')*:
  - `'random'`: Selects $k$ observations at random.
  - `'heuristic'`: Picks $k$ points with smallest sum distance to all other points.
  - `'k-medoids++'`: Smart initialization inspired by k-means++ for better convergence.
  - `'build'`: Deterministic greedy initialization (like PAM). High quality but slower (`O(N^2)`).
  - `array-like`: Custom initial centers of shape `(n_clusters, n_features)`.
