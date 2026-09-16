# scikit-clarans

> A scikit-learn compatible implementation of **CLARANS** and **FastCLARANS** for scalable $k$-medoids clustering.

[![License](https://img.shields.io/github/license/ThienNguyen3001/scikit-clarans)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18366801.svg)](https://doi.org/10.5281/zenodo.18366801)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docs Build](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/docs-build.yml?branch=main&label=Docs%20Build)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/docs-build.yml)
[![Test Suite](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/test_suite.yml?branch=main&label=Test%20Suite)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/test_suite.yml)
[![Quality Check](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/lint_cov_check.yml?branch=main&label=Quality%20Check)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/lint_cov_check.yml)
[![PyPI version](https://img.shields.io/pypi/v/scikit-clarans.svg)](https://pypi.org/project/scikit-clarans/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/194aBBu0wZotnun25dXqlOrDj3HYHKo-a?usp=sharing)

> [!NOTE]
> **Educational & Research Scope**: `scikit-clarans` is currently developed primarily for **learning, algorithmic study, and small-to-medium academic research**. As a pure Python/NumPy implementation, it is clean and accessible for experimentation, but it is **not yet optimized for large-scale Big Data applications** ($N \gg 10^5$).

**scikit-clarans** brings scalable $k$-medoids clustering to Python with a native scikit-learn API. Unlike $k$-means which computes artificial centroids (means), $k$-medoids picks **actual data points** as cluster centers.

### Why k-Medoids over k-Means?
* **Outlier Robust**: Minimizes absolute distance ($\sum d$) rather than squared Euclidean distance ($\sum d^2$), so extreme values won't skew cluster centers.
* **Custom Distance Metrics**: Works with `cosine`, `manhattan`, `euclidean`, or any valid metric—unlike $k$-means which is strictly Euclidean.
* **Directly Interpretable**: Medoids are real observations from your dataset (e.g., representative user profiles, real molecules, exemplary documents).

### CLARANS vs. FastCLARANS: Which one to use?
* **`FastCLARANS` (Recommended for most workloads)**: Uses FastPAM1 delta calculations (Schubert & Rousseeuw, 2021) to evaluate all $k$ medoids at once. Explores $k$ graph edges in the time CLARANS explores one, yielding substantial speedups with $O(n)$ memory.
* **`CLARANS`**: Randomized search (Ng & Han, 2002) with optional distance caching (`cache=True`, default) for fast $O(n)$ swap evaluations, or classic brute-force cost recalculation (`cache=False`).

---

## Features

* **Scikit-Learn Native**: Inherits from `BaseEstimator` and `ClusterMixin`. Plug-and-play in scikit-learn `Pipeline`, `GridSearchCV`, and clustering evaluations.
* **Memory Efficient**: Computes distances on-the-fly ($O(n)$ memory overhead) to easily scale to tens of thousands of samples without blowing up RAM ($O(n^2)$).
* **Flexible Seeding**: Supports multiple initialization strategies (`k-medoids++`, `build`, `random`).

## Installation

Install simply via pip:
```bash
pip install scikit-clarans
```
Or install from source:
```bash
pip install .
```
For development
```bash
pip install -e ".[dev]"
```

## Quick Start
### CLARANS
```python
from clarans import CLARANS
from sklearn.datasets import make_blobs

# 1. Create dummy data
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

# 2. Initialize CLARANS
#    - n_clusters: 5 clusters
#    - num_local: 3 restarts for better quality
#    - init: 'k-medoids++' for smart starting points
#    - cache: True (default) for fast O(n) swap evaluations; False for classic baseline
clarans = CLARANS(n_clusters=5, num_local=3, init='k-medoids++', cache=True, random_state=42)

# 3. Fit
clarans.fit(X)

# 4. Results
print("Medoid Indices:", clarans.medoid_indices_)
print("Labels:", clarans.labels_)
```
### FastCLARANS

**FastCLARANS** implements the faster variant from Schubert & Rousseeuw (2021). It evaluates swaps with all k medoids simultaneously using FastPAM1 delta formulas, exploring k edges of the search graph in the time CLARANS explores one:

```python
from clarans import FastCLARANS

# FastCLARANS computes distances on-the-fly (memory efficient)
# and samples max(250, 2.5% of non-medoid points) per iteration
fast_model = FastCLARANS(n_clusters=5, num_local=3, random_state=42)
fast_model.fit(X)
```

**Key differences from CLARANS:**
- Samples only non-medoid candidates (not medoid-candidate pairs)
- Evaluates swap with all k medoids at once (O(k) speedup per evaluation)
- Memory efficient: O(n) instead of O(n²)

## Examples

This repository includes a number of runnable examples in the `examples/` folder showing common usage patterns and integrations. Run any example with:

```bash
python examples/01_quick_start.py
```

## Documentation

For full API reference and usage guides, please see the [Documentation](https://scikit-clarans.readthedocs.io/en/latest/index.html).

## Contributing

Contributions are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use `scikit-clarans` in your software or research, please cite:

```bibtex
@software{scikit_clarans,
  author       = {Nguyen, Ngoc Thien},
  title        = {scikit-clarans: A Python Library for CLARANS Clustering},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18366801},
  url          = {https://github.com/ThienNguyen3001/scikit-clarans}
}
```

### Academic References

The core algorithms implemented in this package originate from:

* **CLARANS:**
  > Ng, R. T., & Han, J. (2002). *CLARANS: A method for clustering objects for spatial data mining.* IEEE Transactions on Knowledge and Data Engineering, 14(5), 1003-1016. [doi:10.1109/TKDE.2002.1033770](https://doi.org/10.1109/TKDE.2002.1033770)
* **FastCLARANS & FastPAM1:**
  > Schubert, E., & Rousseeuw, P. J. (2021). *Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms.* Information Systems, 101, 101804. [doi:10.1016/j.is.2021.101804](https://doi.org/10.1016/j.is.2021.101804)
* **Seeding & Initialization:**
  > Initialization strategies (`k-medoids++`, `heuristic`, `build`) are adapted from the [scikit-learn-extra KMedoids](https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html) implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
