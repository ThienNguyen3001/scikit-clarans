# scikit-clarans

A scikit-learn compatible implementation of the CLARANS (Clustering Large Applications based on RANdomized Search) algorithm.

## Installation

You can install this package locally:

```bash
pip install .
```

Or in editable mode for development:

```bash
pip install -e .
```

## Usage

```python
from clarans import CLARANS
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=500, centers=3, n_features=2, random_state=42)

# Initialize and fit CLARANS
clarans = CLARANS(n_clusters=3, numlocal=2, maxneighbor=None, random_state=42)
clarans.fit(X)

# Get results
print("Cluster Centers (Indices):", clarans.medoid_indices_)
print("Cluster Centers (Coords):", clarans.cluster_centers_)
print("Labels:", clarans.labels_)
```

## Parameters

- `n_clusters`: The number of clusters to form (default=8).
- `numlocal`: Number of local minima to find (default=2).
- `maxneighbor`: Maximum number of neighbors to examine (default=None, calculated by heuristic).
- `random_state`: Determines random number generation for centroid initialization.

## Compatibility

This library inherits from `sklearn.base.BaseEstimator` and `sklearn.base.ClusterMixin`, making it compatible with scikit-learn pipelines and tools.
