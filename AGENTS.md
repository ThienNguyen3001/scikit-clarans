# AGENTS.md - Complete AI Guidelines, Architecture & Workflows for scikit-clarans

This document is the single, comprehensive source of truth for all AI coding assistants (Antigravity, Cursor, Claude Code, GitHub Copilot) and human contributors working on `scikit-clarans`.

---

## 1. Quick Environment & Build Setup

### Mandatory Virtual Environment
Always execute Python commands with the repository's virtual environment activated:
- **Windows (PowerShell)**: `& .venv\Scripts\Activate.ps1`
- **Linux / macOS (Bash)**: `source .venv/bin/activate`

### The Ninja PATH Gotcha (Windows)
`scikit-clarans` uses `meson-python` with `ninja` for Cython compilation. In editable development mode (`pip install -e ".[dev]"`), imports run through `_scikit_clarans_editable_loader.py`, which triggers `ninja` on the fly to auto-recompile modified C-extensions.
- **Never** call `.venv\Scripts\python.exe` or `.venv\Scripts\pytest.exe` directly without activating the virtual environment or prepending `.venv\Scripts` to `PATH`. Doing so causes:
  `FileNotFoundError: [WinError 2] The system cannot find the file specified` (failure to locate `ninja`).

### Editable Installation
Always use `--no-build-isolation` to avoid slow re-installation of dependencies:
```bash
pip install meson-python ninja cython numpy
pip install -e ".[dev]" --no-build-isolation
```

### Quick Verification
```bash
python -c "import clarans; print('Version:', clarans.__version__, '| HAS_CYTHON:', clarans.HAS_CYTHON)"
```

---

## 2. Hard Architectural & Algorithmic Constraints

### Strict $O(n)$ Memory Footprint
- All algorithms (`CLARANS`, `FastCLARANS`) and utilities must strictly maintain an $O(n)$ working memory footprint.
- **Prohibition**: Never allocate, store, or cache full $O(n^2)$ pairwise distance matrices. Distances must be computed on demand or maintained in minimal working buffers ($O(n)$ or $O(k \cdot n)$).

### Dual Engine Integrity (Cython + Pure Python)
- High-performance C kernels reside in `clarans/_core.pyx`.
- **Type Stubs**: Every function added or modified in `_core.pyx` **must** have matching type annotations in `clarans/_core.pyi`.
- **Pure Python Fallback**: Every C kernel **must** have a functionally identical pure Python/NumPy fallback emitting an `EfficiencyWarning` when `HAS_CYTHON` is `False`. If `_core` fails to import, the library must degrade gracefully while keeping all public APIs functional.
- **Fused Types**: All kernels must support fused numeric types for both `float64` and `float32`.

### Scikit-Learn Estimator Compliance
Both `CLARANS` and `FastCLARANS` inherit from `BaseEstimator`, `ClusterMixin`, and `TransformerMixin`:
1. **Constructor Purity**: `__init__` must only assign arguments directly to instance attributes of the exact same name (e.g. `self.n_clusters = n_clusters`). No parameter validation or transformation may take place in `__init__`.
2. **Validation in `fit()`**: All parameter validation, input checks, and array conversions must occur inside `fit()` via `validate_data()`.
3. **Trailing Underscore**: All attributes estimated from data must end with a single trailing underscore (e.g. `labels_`, `medoid_indices_`, `cluster_centers_`, `inertia_`, `n_iter_`).
4. **Estimator Checks Gate**: Changes must never break `@parametrize_with_checks` in `clarans/tests/test_common.py` (100+ standard sklearn checks).

### Code Style & Quality
- All Python code must pass `flake8 clarans` (`.flake8`: max line length 100, ignore `E203`).
- Static types must verify with `mypy clarans`.

---

## 3. Cython 3.0 Performance & C-Extension Standards

When writing or modifying files in `clarans/_core.pyx`:

1. **Compiler Directives**: Every Cython file must begin with:
   ```cython
   # cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
   ```
2. **Zero GIL Overhead (`with nogil:`)**: All computational loops iterating over `n_samples` ($n$) or `n_clusters` ($k$) must be executed within `with nogil:` blocks. Do not create Python objects, dynamic lists, or call Python C-API functions inside `nogil`.
3. **Fused Types**: Template kernels using:
   ```cython
   ctypedef fused floating:
       double
       float
   ```
4. **Typed Memoryviews**: Use 1D C-contiguous typed memoryviews (`const floating[::1]`, `const intp_t[::1]`). Use `const` on read-only inputs for compiler register optimization.
5. **Buffer Pre-allocation**: In functions evaluated repeatedly (like `fastpam1_delta`), support an optional pre-allocated working buffer (`floating[::1] delta_buf = None`) to eliminate inner heap allocations.
6. **Synchronize `_core.pyi`**: Always update stubs in `clarans/_core.pyi` alongside `_core.pyx`.

---

## 4. Standard Developer & CI Workflows

AI agents and developers should follow these standardized procedures:

### Workflow 1: Build & Compile (`/build`)
```powershell
& .venv\Scripts\Activate.ps1
pip install -e ".[dev]" --no-build-isolation
python -c "import clarans; assert clarans.HAS_CYTHON, 'Cython compilation failed!'"
pytest clarans/tests/test_core.py
```

### Workflow 2: Testing (`/test`)
```powershell
& .venv\Scripts\Activate.ps1
# 1. Test Cython kernels against NumPy reference
pytest clarans/tests/test_core.py -v
# 2. Test Scikit-Learn estimator compliance (100+ checks)
pytest clarans/tests/test_common.py
# 3. Full test suite with coverage
pytest --cov=clarans --cov-report=term-missing clarans/tests/
```

### Workflow 3: Code Quality & Linting (`/lint`)
```powershell
& .venv\Scripts\Activate.ps1
flake8 clarans
mypy clarans
pre-commit run --all-files
```

### Workflow 4: Documentation (`/docs`)
```powershell
& .venv\Scripts\Activate.ps1
pip install -e ".[docs]" --no-build-isolation
sphinx-build -b html docs/source docs/build/html -W --keep-going
```

### Workflow 5: Benchmarking (`/benchmark`)
```powershell
& .venv\Scripts\Activate.ps1
python examples/plot_clarans_vs_fastclarans.py
python examples/plot_runtime_scaling.py
python examples/plot_cost_evaluation_strategy.py
```

### Workflow 6: Examples Smoke-Test (`/examples`)
Run all 19 gallery scripts with headless `MPLBACKEND=Agg`:
```powershell
& .venv\Scripts\Activate.ps1
$env:MPLBACKEND = "Agg"
Get-ChildItem examples/plot_*.py | ForEach-Object {
    Write-Host "Testing $($_.Name)..." -ForegroundColor Cyan
    python $_.FullName
}
```

### Workflow 7: Pre-flight Gatekeeper (`/preflight`)
Run before opening a PR or tagging a release:
1. Rebuild Cython: `pip install -e ".[dev]" --no-build-isolation`
2. Lint: `flake8 clarans` && `mypy clarans`
3. Sklearn check: `pytest clarans/tests/test_common.py`
4. Full tests: `pytest --cov=clarans clarans/tests/`
5. Pre-commit: `pre-commit run --all-files`

### Workflow 8: Package Release (`/release`)
1. Verify identical version strings in `pyproject.toml`, `meson.build`, `clarans/__init__.py`.
2. Run `/preflight`.
3. Clean build folders: `Remove-Item -Recurse -Force dist, build`
4. Build packages: `python -m build --sdist --wheel`
5. Validate packages: `twine check dist/*`
6. Upload to PyPI: `twine upload dist/*`
7. Tag git release: `git tag -a "v$ver" -m "Release version $ver"` && `git push origin "v$ver"`

---

## 5. Machine Learning & Algorithm Usage Guide

### CLARANS vs. FastCLARANS Decision Matrix

| Criterion | FastCLARANS (Recommended) | CLARANS (Classic) |
| :--- | :--- | :--- |
| **Swap Evaluation** | Evaluates swaps across all $k$ medoids in a single pass using FastPAM1 delta calculations. | Evaluates a single random `(medoid, candidate)` pair per step. |
| **Runtime** | $O(k)$ fewer distance queries; substantially faster for $k \ge 5$. | Slower for large $k$; tests pairs randomly. |
| **Search Space** | Evaluates $k$ swap combinations per node; avoids early stagnation. | Prone to premature exit if small integer `max_neighbors` is set. |
| **Best For** | Tabular datasets, production pipelines, large data clustering. | Algorithmic comparison baselines, reproducing Ng & Han (2002). |

### Key Hyperparameters
- `n_clusters`: Number of clusters ($k$).
- `num_local`: Default `2` to `5`. Number of local search restarts from different random points.
- `max_neighbors`: Default `'auto'` ($\max(250, 1.25\% \times k(n-k))$). **Never pass small integers (e.g. 40)** as it causes early termination.
- `init`: `'k-medoids++'` (recommended default), `'random'`, `'heuristic'`, or `'build'`.
- `metric`: Supports `'euclidean'`, `'manhattan'`, `'cosine'`, `'precomputed'`, or custom callables.

### Core Usage Patterns

#### 1. Pipeline Integration
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

#### 2. Predict & Transform Unseen Samples
```python
# Predict nearest medoid for new points
labels_new = clusterer.predict(X_new)

# Transform maps points to distances to all k medoids
distances = clusterer.transform(X_new)  # shape (n_samples, n_clusters)
```

#### 3. Precomputed Distance Matrix
```python
from sklearn.metrics import pairwise_distances
from clarans import FastCLARANS

D = pairwise_distances(X, metric="cosine")
model = FastCLARANS(n_clusters=4, metric="precomputed", random_state=42)
model.fit(D)
print("Medoid Indices:", model.medoid_indices_)
```

#### 4. Sparse Matrix Inputs (`scipy.sparse`)
`scikit-clarans` automatically routes sparse inputs to `DistanceMetric` without densifying into full matrices:
```python
from scipy.sparse import csr_matrix
from clarans import FastCLARANS

X_sparse = csr_matrix(X)
model = FastCLARANS(n_clusters=5, metric="euclidean", random_state=42)
model.fit(X_sparse)
```
