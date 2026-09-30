# scikit-clarans

> Scikit-learn compatible implementation of CLARANS and FastCLARANS for $k$-medoids clustering.

[![License](https://img.shields.io/github/license/ThienNguyen3001/scikit-clarans)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18366801.svg)](https://doi.org/10.5281/zenodo.18366801)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docs Build](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/docs-build.yml?branch=main&label=Docs%20Build)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/docs-build.yml)
[![Test Suite](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/test_suite.yml?branch=main&label=Test%20Suite)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/test_suite.yml)
[![Quality Check](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/lint_cov_check.yml?branch=main&label=Quality%20Check)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/lint_cov_check.yml)
[![PyPI version](https://img.shields.io/pypi/v/scikit-clarans.svg)](https://pypi.org/project/scikit-clarans/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/194aBBu0wZotnun25dXqlOrDj3HYHKo-a?usp=sharing)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/scikit-clarans?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/scikit-clarans)

> [!NOTE]
> This library is designed for coursework, algorithm study, and academic research. It uses Cython C-extensions with a pure Python fallback and maintains an $O(n)$ memory footprint instead of allocating an $O(n^2)$ pairwise distance matrix.

scikit-clarans implements $k$-medoids clustering in Python using the scikit-learn estimator interface. While $k$-means calculates artificial cluster centroids, $k$-medoids picks actual data points from the dataset as cluster centers.

### Why use k-medoids?
* Outlier resistance: Medoids minimize absolute distances ($\sum d$) rather than squared Euclidean distances ($\sum d^2$), keeping centers stable when the dataset contains extreme values.
* Custom distance metrics: Supports metrics such as Manhattan, cosine, or precomputed distances, whereas standard $k$-means requires Euclidean space.
* Direct interpretability: Every cluster center corresponds to a real record in the input data.

### Choosing between CLARANS and FastCLARANS
* FastCLARANS: Evaluates swaps across all $k$ medoids in a single pass using FastPAM1 delta calculations (Schubert & Rousseeuw, 2021). It uses $O(n)$ memory and is faster for most workloads.
* CLARANS: The classic randomized search algorithm from Ng & Han (2002). It supports delta cost evaluations with cached distances as well as brute-force recalculation.

---

## Features

* Scikit-learn compatibility: Extends `BaseEstimator` and `ClusterMixin` for use in `Pipeline`, `GridSearchCV`, and model evaluation workflows.
* C acceleration: Core delta computations run in Cython with a pure Python and NumPy fallback.
* Distance routing: Automatically routes distance calculations to SciPy `cdist` for dense arrays, scikit-learn `DistanceMetric` for sparse matrices, or `pairwise_distances`.
* Low memory overhead: Computes distances as needed with $O(n)$ working memory instead of storing a full $O(n^2)$ matrix.
* Initialization options: Supports `k-medoids++`, `build`, `heuristic`, uniform random sampling, or user-supplied medoid indices.

## Installation

Install from PyPI:
```bash
pip install scikit-clarans
```

Install from source:
```bash
git clone https://github.com/ThienNguyen3001/scikit-clarans.git
cd scikit-clarans
pip install .
```

For development:
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
clarans = CLARANS(n_clusters=5, num_local=3, init='k-medoids++', cost_evaluation='delta', random_state=42)

# 3. Fit
clarans.fit(X)

# 4. Results
print("Medoid Indices:", clarans.medoid_indices_)
print("Labels:", clarans.labels_)
```

### FastCLARANS

FastCLARANS evaluates swaps across all $k$ medoids simultaneously using the FastPAM1 formulation from Schubert & Rousseeuw (2021):

```python
from clarans import FastCLARANS

fast_model = FastCLARANS(n_clusters=5, num_local=3, random_state=42)
fast_model.fit(X)
```

Differences from CLARANS:
- Samples non-medoid candidates instead of medoid-candidate pairs.
- Evaluates swaps against all $k$ medoids in a single pass ($O(k)$ fewer distance queries).
- Operates in $O(n)$ memory instead of $O(n^2)$.

## Examples

Runnable scripts are located in the `examples/` directory:

```bash
python examples/plot_quick_start.py
```

## Documentation

Full API documentation and guides are available at https://scikit-clarans.readthedocs.io.

## Contributing

Contributions and bug reports are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and testing instructions.

## Citation

If you use `scikit-clarans` in your research or software, please cite:

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

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
