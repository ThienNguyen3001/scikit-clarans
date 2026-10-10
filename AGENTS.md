# Developer and Agent Guidelines for scikit-clarans

Guidelines, architecture rules, and workflows for working on `scikit-clarans`.

---

## 1. Environment and build setup

### Virtual environment
Activate the repository virtual environment before running Python commands:
- **Windows (PowerShell)**: `& .venv\Scripts\Activate.ps1`
- **Linux / macOS (Bash)**: `source .venv/bin/activate`

### Ninja PATH on Windows
`scikit-clarans` uses `meson-python` with `ninja` for Cython compilation. In editable development mode (`pip install -e ".[dev]"`), imports run through `_scikit_clarans_editable_loader.py`, which invokes `ninja` on the fly to recompile modified C extensions.

Do not call `.venv\Scripts\python.exe` or `.venv\Scripts\pytest.exe` directly without activating the virtual environment or prepending `.venv\Scripts` to `PATH`. Otherwise, the system fails to locate `ninja` and raises:
`FileNotFoundError: [WinError 2] The system cannot find the file specified`

### Editable installation
Use `--no-build-isolation` to avoid reinstalling dependencies on every build:
```bash
pip install meson-python ninja cython numpy
pip install -e ".[dev]" --no-build-isolation
```

### Verification
```bash
python -c "import clarans; print('Version:', clarans.__version__, '| HAS_CYTHON:', clarans.HAS_CYTHON)"
```

---

## 2. Architecture and algorithmic constraints

### Memory footprint
- `CLARANS`, `FastCLARANS`, and helper utilities must stay within an $O(n)$ working memory footprint.
- Never allocate, store, or cache full $O(n^2)$ pairwise distance matrices in the clustering search loops. Compute distances on demand or keep them in minimal buffers sized $O(n)$ or $O(k \cdot n)$. Note that greedy initializations (`init='build'` and `init='heuristic'`) compute full pairwise distance matrices ($O(n^2)$) by definition.

### Cython and pure Python engines
- Cython kernels live in `clarans/_core.pyx`:
  1. `clarans_delta`: Single swap delta calculation for CLARANS.
  2. `fastpam1_delta`: Simultaneous FastPAM1 delta calculation across all $k$ medoids for FastCLARANS.
  3. `update_cache_2min`: Finds nearest ($d_1$) and second-nearest ($d_2$) medoids and distances.
  4. `pam_build_step`: Greedy distance reduction for PAM BUILD initialization.
  5. `kmedoids_pp_trials`: Evaluates candidate potentials across local trials for $k$-medoids++.
  6. `is_matrix_symmetric`: Fast C symmetry validation for precomputed distance matrices.
- Every function in `clarans/_core.pyx` needs matching type annotations in `clarans/_core.pyi`.
- Every Cython kernel must have an equivalent pure Python or NumPy fallback that raises an `EfficiencyWarning` when `HAS_CYTHON` is `False`. If `_core` fails to import, all public APIs must remain usable.
- Kernels must support fused numeric types for both `float64` and `float32`.

### Scikit-learn estimator requirements
Both `CLARANS` and `FastCLARANS` inherit from `BaseEstimator`, `ClusterMixin`, and `TransformerMixin`:
1. Assign arguments directly to instance attributes of the exact same name in `__init__` (such as `self.n_clusters = n_clusters`). Do not validate or transform parameters in `__init__`.
2. Run all parameter validation, input checks, and array conversions inside `fit()` using `validate_data()`.
3. Name all fitted attributes with a single trailing underscore:
   - `labels_`: Cluster labels for each training sample.
   - `medoid_indices_`: Indices of selected medoids in training data.
   - `cluster_centers_`: Coordinates of medoids (set to `None` if `metric='precomputed'`).
   - `inertia_`: Total sum of distances from samples to closest medoids.
   - `max_neighbors_`: Effective sampling budget per local search restart.
   - `n_neighbors_`: Total neighborhood size in search space ($k(n-k)$ for CLARANS, $n-k$ for FastCLARANS).
   - `n_iter_`: Total candidate neighbors evaluated across all restarts.
   - `n_swaps_`: Total successful medoid swaps performed.
   - `n_features_in_`: Number of features seen during `fit()` (dense/sparse mode).
4. Implement standard estimator methods: `fit(X)`, `predict(X)`, `transform(X)`, `fit_predict(X)`, `fit_transform(X)`, `score(X)`, and `get_feature_names_out(input_features=None)`.
5. Ensure all checks in `clarans/tests/test_common.py` pass via `@parametrize_with_checks`.

### Code style and quality
- Format code with `black` (`[tool.black]`: max line length 100, py39-py313 targets).
- Python code must pass `flake8 clarans` (`.flake8`: max line length 100, ignore `E203`).
- Static types must verify with `mypy clarans`.
- Pre-commit hooks (`black`, `flake8`, whitespace fixers) configured in `.pre-commit-config.yaml`.

---

## 3. Cython guidelines

When writing or modifying files in `clarans/_core.pyx`:

1. Include these compiler directives at the top of every Cython file:
   ```cython
   # cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
   ```
2. Run loops over `n_samples` ($n$) or `n_clusters` ($k$) inside `with nogil:` blocks. Do not allocate Python objects, create lists, or call Python C-API functions inside `nogil`.
3. Support fused floating types:
   ```cython
   ctypedef fused floating:
       double
       float
   ```
4. Use 1D C-contiguous typed memoryviews (`const floating[::1]`, `const intp_t[::1]`). Mark read-only inputs `const`.
5. Support optional pre-allocated working buffers (such as `floating[::1] delta_buf = None` in `fastpam1_delta`) to avoid repeated heap allocations in hot loops.
6. Keep type stubs in `clarans/_core.pyi` synchronized with `clarans/_core.pyx`.

---

## 4. Workflows

### Build (`/build`)
```powershell
& .venv\Scripts\Activate.ps1
pip install -e ".[dev]" --no-build-isolation
python -c "import clarans; assert clarans.HAS_CYTHON, 'Cython compilation failed!'"
pytest clarans/tests/test_core.py
```

### Test (`/test`)
```powershell
& .venv\Scripts\Activate.ps1
# 1. Test Cython kernels against NumPy reference (float64 and float32)
pytest clarans/tests/test_core.py -v
# 2. Test Scikit-Learn estimator compliance (100+ checks)
pytest clarans/tests/test_common.py
# 3. Full test suite with coverage
pytest --cov=clarans --cov-report=term-missing clarans/tests/
```

#### Test suite structure (`clarans/tests/`):
- `test_core.py`: Direct unit tests comparing Cython kernels against NumPy references across fused types.
- `test_common.py`: Estimator contract verification via scikit-learn's `parametrize_with_checks`.
- `test_ecosystem.py`: Scikit-learn integration (`Pipeline`, `GridSearchCV`, `clone`, `fit_predict`, pickle/joblib serialization, Fortran order, strided slices, custom callable metrics).
- `test_initialization.py`: Unit tests for `k-medoids++`, `random`, `heuristic`, `build`, and explicit center arrays.
- `test_clarans.py` & `test_fast_clarans.py`: Core functional tests, distance caching, and parameter validation.
- `test_regressions.py` & `test_robustness.py`: Numerical stability, memory leak cleanup, overflow handling, and bug regression prevention.

### Lint (`/lint`)
```powershell
& .venv\Scripts\Activate.ps1
black --check clarans examples
flake8 clarans
mypy clarans
pre-commit run --all-files
```

### Documentation (`/docs`)
```powershell
& .venv\Scripts\Activate.ps1
pip install -e ".[docs]" --no-build-isolation
sphinx-build -b html docs/source docs/build/html -W --keep-going
```

### Benchmark (`/benchmark`)
```powershell
& .venv\Scripts\Activate.ps1
# 1. Gallery benchmarks
python examples/plot_clarans_vs_fastclarans.py
python examples/plot_runtime_scaling.py
python examples/plot_cost_evaluation_strategy.py

# 2. Interactive and telemetry benchmarks
# - benchmarks/benchmark_clarans_colab.ipynb (Large-scale Colab benchmarking)
# - unsupervised-anomaly-detection.ipynb (Unsupervised anomaly detection on Numenta NAB sensor telemetry)
```

### Examples (`/examples`)
Run all 20 gallery scripts with headless `MPLBACKEND=Agg`:
```powershell
& .venv\Scripts\Activate.ps1
$env:MPLBACKEND = "Agg"
Get-ChildItem examples/plot_*.py | ForEach-Object {
    Write-Host "Testing $($_.Name)..." -ForegroundColor Cyan
    python $_.FullName
}
```

### JOSS Paper (`/paper`)
The `paper/` directory contains manuscript files for the *Journal of Open Source Software* (JOSS):
- `paper/paper.md`: Main manuscript.
- `paper/paper.bib`: BibTeX bibliography.
- `paper/generate_results_plot.py`: Generates the benchmark figure `paper/results.png`.
- Compile PDF locally with Inara (requires Docker):
  ```bash
  docker run --rm -v "${PWD}/paper:/data" -w /data openjournals/inara -o pdf,crossref paper.md
  ```
- Automated builds trigger on push to branch `paper` via `.github/workflows/draft-pdf.yml`.

### Pre-flight checks (`/preflight`)
Run before opening a pull request or tagging a release:
1. Rebuild Cython: `pip install -e ".[dev]" --no-build-isolation`
2. Lint: `black --check clarans examples` && `flake8 clarans` && `mypy clarans`
3. Sklearn check: `pytest clarans/tests/test_common.py`
4. Full tests: `pytest --cov=clarans clarans/tests/`
5. Pre-commit: `pre-commit run --all-files`

### CI/CD Workflows (`.github/workflows/`)
1. `test_suite.yml`: Cross-platform test matrix across Ubuntu (Python 3.9–3.13), Windows (3.9, 3.13 with MSVC), and macOS (3.9, 3.13).
2. `lint_cov_check.yml`: Code quality checks (`flake8`, `mypy`, `pytest` coverage XML) on Python 3.12.
3. `docs-build.yml`: Builds Sphinx HTML documentation with `-W --keep-going` on Python 3.12.
4. `draft-pdf.yml`: Compiles JOSS draft paper PDF on pushes to branch `paper`.
5. `pypi-publish.yml`: Compiles multi-platform binary wheels (`cibuildwheel`) and sdist, publishing to PyPI via OIDC trusted publishing upon release publication or workflow dispatch.

### Release (`/release`)

#### Step A: Local dry-run validation
Verify locally that sdist and wheels compile cleanly and metadata passes twine validation without uploading:
1. Check that version strings match in `pyproject.toml`, `meson.build`, and `clarans/__init__.py`.
2. Run `/preflight` to verify tests, linters, and sklearn checks pass.
3. Clean previous build folders: `Remove-Item -Recurse -Force dist, build`
4. Build source archive and wheel: `python -m build --sdist --wheel`
5. Validate archive metadata: `twine check dist/*`

#### Step B: Publish via GitHub Actions
Publishing to PyPI is handled by GitHub Actions (`.github/workflows/pypi-publish.yml`), which uses `cibuildwheel` to build binary wheels across Ubuntu, Windows, and macOS using OIDC trusted publishing. Do not run `twine upload` locally for production releases.

1. Tag and push the new release:
   ```powershell
   $ver = python -c "import clarans; print(clarans.__version__)"
   git tag -a "v$ver" -m "Release version $ver"
   git push origin "v$ver"
   ```
2. **Publish GitHub Release**: `pypi-publish.yml` triggers on `release: [published]` or `workflow_dispatch` (pushing the git tag alone does *not* trigger PyPI publishing). Create and publish the GitHub release from the tag:
   ```powershell
   gh release create "v$ver" --generate-notes
   ```
   Or create the release manually via GitHub Web UI (`Releases -> Draft a new release -> Choose tag v$ver -> Publish release`).
3. Monitor build progress in GitHub Actions: `https://github.com/ThienNguyen3001/scikit-clarans/actions/workflows/pypi-publish.yml`

#### Step C: TestPyPI staging
To test package installation in an isolated sandbox before tagging:
```powershell
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple scikit-clarans
```

---

## 5. Algorithm usage and guide

### Comparing CLARANS and FastCLARANS

| Criterion | FastCLARANS (Recommended) | CLARANS (Classic) |
| :--- | :--- | :--- |
| Swap evaluation | Evaluates swaps across all $k$ medoids in a single pass using FastPAM1 delta calculations. | Evaluates a single random `(medoid, candidate)` pair per step. |
| Runtime | $O(k)$ fewer distance queries; faster for $k \ge 5$. | Slower for large $k$ because pairs are sampled randomly. |
| Search space | Samples non-medoid nodes ($V \setminus M$) and evaluates $k$ edges simultaneously. | Samples individual edges $(m, c)$ uniformly at random. |
| Auto budget | $\max(1, \lfloor 250 / k \rfloor, \lfloor 0.025(n - k) \rfloor)$ non-medoid candidates. | $\max(250, \lfloor 0.0125 \cdot k(n - k) \rfloor)$ edge pairs. |
| Typical use | Tabular datasets, production pipelines, and larger datasets. | Baseline comparisons and reproducing Ng & Han (2002). |

### Key hyperparameters
- `n_clusters`: Number of clusters ($k$).
- `num_local`: Number of local search restarts (typically 2 to 5).
- `max_neighbors`: Neighbor sampling budget per local search:
  - `FastCLARANS`: Default `'auto'` sets $\max(1, \lfloor 250 / k \rfloor, \lfloor 0.025(n - k) \rfloor)$ non-medoid candidates (Schubert & Rousseeuw, 2021).
  - `CLARANS`: Default `'auto'` sets $\max(250, \lfloor 0.0125 \cdot k(n - k) \rfloor)$ edge evaluations (Ng & Han, 2002).
  - Avoid small integer constants (like 40), which trigger premature stagnation.
- `cost_evaluation`: Strategy for swap evaluation (`CLARANS` only):
  - `'delta'` (default): FastPAM1-style nearest and second-nearest distance tracking in $O(n \cdot d)$.
  - `'brute_force'`: Recalculates full clustering cost in $O(n \cdot k \cdot d)$ per candidate (classic Ng & Han).
  - (`FastCLARANS` always uses `'delta'`).
- `init`: Initialization strategy (`'k-medoids++'`, `'random'`, `'heuristic'`, `'build'`, or explicit array-like).
- `metric`: Distance metric (`'euclidean'`, `'manhattan'`, `'cosine'`, `'chebyshev'`, `'precomputed'`, or a callable).
- `metric_params`: Optional dictionary of keyword arguments for the metric (such as `{'VI': VI}` for `'mahalanobis'`).
- `verbose`: Verbosity mode:
  - `0`: Silent (default).
  - `1`: Progress per restart (cost, elapsed time, swaps, evaluations).
  - `>=2`: Details per successful medoid swap.
- `random_state`: Integer or `RandomState` for deterministic results across runs.

### Usage patterns

#### 1. Pipeline integration
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from clarans import FastCLARANS

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clusterer', FastCLARANS(n_clusters=5, num_local=3, random_state=42))
])
pipeline.fit(X)
clusterer = pipeline.named_steps['clusterer']
print("Medoid Indices:", clusterer.medoid_indices_)
print("Cluster Centers:", clusterer.cluster_centers_)
```

#### 2. Predict and transform new samples
```python
# Predict nearest medoid for new points
labels_new = clusterer.predict(X_new)

# Transform maps points to distances to all k medoids
distances = clusterer.transform(X_new)  # shape (n_samples, n_clusters)

# Feature names out for downstream sklearn transformers
feature_names = clusterer.get_feature_names_out()
```

#### 3. Precomputed distance matrix
```python
from sklearn.metrics import pairwise_distances
from clarans import FastCLARANS

D = pairwise_distances(X, metric="cosine")
model = FastCLARANS(n_clusters=4, metric="precomputed", random_state=42)
model.fit(D)
print("Medoid Indices:", model.medoid_indices_)
```

#### 4. Sparse matrix inputs (`scipy.sparse`)
`scikit-clarans` routes sparse inputs to `DistanceMetric` without densifying the full matrix:
```python
from scipy.sparse import csr_matrix
from clarans import FastCLARANS

X_sparse = csr_matrix(X)
model = FastCLARANS(n_clusters=5, metric="euclidean", random_state=42)
model.fit(X_sparse)
```

---

## 6. Distance metric routing

`scikit-clarans` routes distance evaluations based on input type and metric:
1. `'precomputed'`: Validates matrix symmetry with `_core.is_matrix_symmetric`. In precomputed mode, `cluster_centers_` is set to `None` because coordinate data does not exist. Transposes row/column slices automatically if precomputed matrix is asymmetric.
2. `cdist` (SciPy): Used for dense arrays with standard metrics (`euclidean`, `cityblock`, `cosine`, `chebyshev`). Metric aliases mapped in `_SCIPY_METRIC_MAP` translate `manhattan` to `cityblock`, `l1` to `cityblock`, `l2` to `euclidean`, `infinity` to `chebyshev`, and `sokalmichener` to `matching`.
3. `DistanceMetric` (scikit-learn): Used for sparse inputs (`scipy.sparse.csr_matrix` and `csc_matrix`) to compute row-wise distances without densifying the matrix.
4. `pairwise`: Fallback for custom callable metrics (`metric=func`).
5. `nan_euclidean`: Computes pairwise Euclidean distances over non-missing feature subsets when inputs contain NaNs.
6. Special requirements: Metric `'mahalanobis'` requires the inverse covariance matrix `VI` passed in `metric_params`.

---

## 7. Initialization methods (`_initialization.py`)

- `'k-medoids++'` (default): Probabilistic $D^2$-weighting adapted from Arthur & Vassilvitskii (2007). Evaluates $2 + \ln(k)$ local trials via `_core.kmedoids_pp_trials`.
- `'random'`: Uniform random sample of $k$ distinct points.
- `'heuristic'`: Selects points with the lowest sum of distances to all other samples. Computes full $O(n^2)$ pairwise distances.
- `'build'`: Greedy BUILD phase from Kaufman & Rousseeuw (1990). Selects an initial center minimizing total distance, then adds medoids that maximize distance reduction. Computes $O(n^2)$ distances, accelerated by `_core.pam_build_step`.
- Explicit array (`init=array_like`): Snaps supplied coordinates to nearest unique training samples and backfills duplicates with random points.
- **Deterministic cache optimization** (`_prepare_initial_medoids`): For deterministic initializers (`'build'`, `'heuristic'`, explicit array), initial medoids are calculated once before the `num_local` loop rather than recalculated every restart, issuing a `UserWarning` if `num_local > 1`.

---

## 8. Numerical invariants and edge cases

These safeguards are tested in `test_regressions.py` and `test_robustness.py`:

1. Delta tolerance: Accept swaps only when `total_delta < _delta_tolerance(cost)`, where `_delta_tolerance(cost) = -max(1e-16, 1e-12 * abs(cost))`. This prevents infinite swap loops caused by floating-point rounding when delta is near zero.
2. Scikit-learn `score(X)`: Returns negative inertia (`-self.inertia_`) so higher values indicate better fits in `GridSearchCV`.
3. Memory cleanup (`_precomputed_source`): Cleans up `self._precomputed_source` in a `finally` block if `fit()` raises an exception on precomputed matrices.
4. Scikit-learn compatibility: Supports both modern `__sklearn_tags__` (scikit-learn >= 1.6) and legacy `_more_tags()`.
5. Numerical overflow: Verifies costs with `np.isfinite()`. If values exceed float64 limits ($> 10^{160}$) and square Euclidean distances overflow to infinity, raises a `ValueError`.
6. Thread safety for warnings: Cython fallback warnings (`_warn_cython_unavailable()`) use double-checked locking with `threading.Lock()` to avoid duplicate warnings under multithreading.
7. Precomputed matrix dimension flexibility: `predict()` and `transform()` accept either full test-to-train pairwise distance matrices of shape $(n_{test}, n_{train})$ or subset matrices of shape $(n_{test}, k)$ matching medoids.
8. Candidate pool management: `_update_candidate_pool` efficiently maintains the boolean mask of available non-medoid candidates after swaps without re-filtering the entire dataset.

---

## 9. Citations

Publications and citation details:

- **CLARANS**:
  > Ng, R. T., & Han, J. (2002). *CLARANS: A method for clustering objects for spatial data mining.* IEEE TKDE, 14(5), 1003-1016. [doi:10.1109/TKDE.2002.1033770](https://doi.org/10.1109/TKDE.2002.1033770)
- **FastCLARANS & FastPAM1**:
  > Schubert, E., & Rousseeuw, P. J. (2021). *Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms.* Information Systems, 101, 101804. [doi:10.1016/j.is.2021.101804](https://doi.org/10.1016/j.is.2021.101804)
- **Library Citation**:
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

