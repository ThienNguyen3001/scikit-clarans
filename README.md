# scikit-clarans

> A scikit-learn compatible implementation of the **CLARANS** (Clustering Large Applications based on RANdomized Search) algorithm.

[![License](https://img.shields.io/github/license/ThienNguyen3001/scikit-clarans)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18366801.svg)](https://doi.org/10.5281/zenodo.18366801)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docs Build](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/docs-build.yml?branch=main&label=Docs%20Build)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/docs-build.yml)
[![Test Suite](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/test_suite.yml?branch=main&label=Test%20Suite)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/test_suite.yml)
[![Quality Check](https://img.shields.io/github/actions/workflow/status/ThienNguyen3001/scikit-clarans/lint_cov_check.yml?branch=main&label=Quality%20Check)](https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/lint_cov_check.yml)
[![PyPI version](https://img.shields.io/pypi/v/scikit-clarans.svg)](https://pypi.org/project/scikit-clarans/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JdgVaZcbS1uwY7kPQZM8DtX97R9ga31d?usp=sharing)

**CLARANS** acts as a bridge between the high quality of **PAM (Partition Around Medoids)** and the speed required for large datasets. By using randomized search instead of exhaustive search, it finds high-quality medoids efficiently without exploring the entire graph of solutions.

---

## Features

*   **Scikit-Learn Native**: Use it just like `KMeans` or `DBSCAN`. Drop-in compatibility for pipelines and cross-validation.
*   **Scalable**: Designed to handle datasets where standard PAM/k-medoids is too slow.
*   **Flexible**: Choose from multiple initialization strategies (`k-medoids++`, `build`, etc.) and distance metrics (`euclidean`, `manhattan`, `cosine`, etc.).

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
#    - numlocal: 3 restarts for better quality
#    - init: 'k-medoids++' for smart starting points
clarans = CLARANS(n_clusters=5, numlocal=3, init='k-medoids++', random_state=42)

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
# and samples 2.5% of non-medoid points per iteration
fast_model = FastCLARANS(n_clusters=5, numlocal=3, random_state=42)
fast_model.fit(X)
```

**Key differences from CLARANS:**
- Samples only non-medoid candidates (not medoid-candidate pairs)
- Evaluates swap with all k medoids at once (O(k) speedup per evaluation)
- Memory efficient: O(n) instead of O(n²)

## Examples

This repository includes a number of runnable examples in the `examples/` folder showing common usage patterns, integrations and a Jupyter notebook (`examples/clarans_examples.ipynb`) with many interactive recipes. Run any example with::

    python examples/01_quick_start.py

## Documentation

For full API reference and usage guides, please see the [Documentation](https://scikit-clarans.readthedocs.io/en/latest/index.html).

## Contributing

Contributions are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use `scikit-clarans` in your research, please cite:

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
