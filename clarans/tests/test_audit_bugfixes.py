"""Tests for bugfixes discovered during deep codebase audit.

Covers:
1. Direction of asymmetric precomputed distance in k-medoids++ initialization.
2. Support for tuple, range, and general array-like init arguments.
3. is_matrix_symmetric handling of NaNs, floating-point tolerances, and scale.
4. Model score() implementation and seamless GridSearchCV default compatibility.
5. End-to-end clustering with metric="nan_euclidean" on missing data.
6. Precomputed Fortran-order symmetric matrices preserving Cython acceleration.
7. Buffer safety check in fastpam1_delta.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import pairwise_distances

from clarans import CLARANS, FastCLARANS
from clarans._initialization import initialize_k_medoids_plus_plus

try:
    from clarans._core import fastpam1_delta, is_matrix_symmetric
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class TestAuditBugfixes:
    def test_kmedoids_pp_asymmetric_precomputed_direction(self):
        """Verify k-medoids++ uses column indexing (sample -> medoid) on asymmetric matrix."""
        # Distance from sample i to medoid j is D[i, j].
        # If medoid 0 is selected:
        # D[1, 0] = 100.0 (sample 1 is far from medoid 0)
        # D[2, 0] = 0.1   (sample 2 is close to medoid 0)
        # However, the row D[0, 1] = 0.1 and D[0, 2] = 100.0 (asymmetric!)
        D = np.array([
            [0.0, 0.1, 100.0, 100.0],  # row 0
            [100.0, 0.0, 50.0, 50.0],  # row 1
            [0.1, 50.0, 0.0, 50.0],    # row 2
            [0.1, 50.0, 50.0, 0.0],    # row 3
        ], dtype=np.float64)

        # Run initialization for 2 clusters across 50 seeds
        selected_candidates = []
        for seed in range(50):
            r = np.random.RandomState(seed)
            medoids = initialize_k_medoids_plus_plus(
                D, n_clusters=2, random_state=r, metric="precomputed"
            )
            if medoids[0] == 0:
                selected_candidates.append(medoids[1])

        # Sample 1 should be selected far more frequently than sample 2 because D[1, 0] == 100
        count_1 = selected_candidates.count(1)
        count_2 = selected_candidates.count(2)
        assert count_1 > count_2

    def test_init_accepts_tuple_and_range(self):
        """Verify init accepts 2D tuple for feature space, and 1D tuple/range for precomputed."""
        X, _ = make_blobs(n_samples=30, centers=2, n_features=2, random_state=42)

        # 1. 2D Tuple of coordinates for feature matrix (sklearn KMeans convention)
        init_tuples = ((float(X[0, 0]), float(X[0, 1])), (float(X[1, 0]), float(X[1, 1])))
        m1 = CLARANS(n_clusters=2, init=init_tuples, num_local=1, random_state=42).fit(X)
        assert len(m1.medoid_indices_) == 2
        m1_fast = FastCLARANS(
            n_clusters=2, init=init_tuples, num_local=1, random_state=42
        ).fit(X)
        assert len(m1_fast.medoid_indices_) == 2

        # 2. 1D Tuple of indices for precomputed matrix
        D = pairwise_distances(X)
        m2 = CLARANS(
            n_clusters=2, init=(0, 1), metric="precomputed", num_local=1, random_state=42
        ).fit(D)
        assert len(m2.medoid_indices_) == 2
        m2_fast = FastCLARANS(
            n_clusters=2, init=(0, 1), metric="precomputed", num_local=1, random_state=42
        ).fit(D)
        assert len(m2_fast.medoid_indices_) == 2

        # 3. 1D Range of indices for precomputed matrix
        m3 = CLARANS(
            n_clusters=2, init=range(2), metric="precomputed", num_local=1, random_state=42
        ).fit(D)
        assert len(m3.medoid_indices_) == 2
        m3_fast = FastCLARANS(
            n_clusters=2, init=range(2), metric="precomputed", num_local=1, random_state=42
        ).fit(D)
        assert len(m3_fast.medoid_indices_) == 2

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython core not available")
    def test_is_matrix_symmetric_nan_and_scale(self):
        """Verify is_matrix_symmetric handles NaNs correctly and supports rtol/atol."""
        # 1. NaN presence should immediately report non-symmetric (False)
        nan_matrix = np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float64)
        assert not is_matrix_symmetric(nan_matrix, 2)

        nan_asym = np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float64)
        assert not is_matrix_symmetric(nan_asym, 2)

        # 2. Float32 large scale values within rtol tolerance
        large_val = 1e6
        D = np.full((5, 5), large_val, dtype=np.float64)
        np.fill_diagonal(D, 0.0)
        # Perturb by 2.0: diff is 2.0, threshold is 1e-8 + 1e-5 * 1e6 = 10.0 -> symmetric!
        D[0, 1] += 2.0
        assert is_matrix_symmetric(D, 5, rtol=1e-5, atol=1e-8)

        # Perturb by 20.0: diff is 20.0 > 10.0 -> not symmetric!
        D[0, 1] += 18.0
        assert not is_matrix_symmetric(D, 5, rtol=1e-5, atol=1e-8)

    def test_score_method_and_gridsearch_default(self):
        """Verify score(X) returns negative inertia and works seamlessly in GridSearchCV."""
        X, _ = make_blobs(n_samples=60, centers=3, n_features=2, random_state=42)

        model = CLARANS(n_clusters=3, random_state=42)

        # Calling score before fit should raise NotFittedError
        with pytest.raises(NotFittedError):
            model.score(X)

        model.fit(X)
        score_val = model.score(X)

        # score(X) on training data should equal -inertia_
        assert np.isclose(score_val, -model.inertia_)

        # score on unseen data should return negative float
        X_test, _ = make_blobs(n_samples=20, centers=3, n_features=2, random_state=99)
        test_score = model.score(X_test)
        assert isinstance(test_score, float)
        assert test_score < 0

        # FastCLARANS score test
        fast_model = FastCLARANS(n_clusters=3, random_state=42).fit(X)
        assert np.isclose(fast_model.score(X), -fast_model.inertia_)

        # GridSearchCV without specifying scoring parameter
        grid = GridSearchCV(
            CLARANS(random_state=42, num_local=1),
            param_grid={"n_clusters": [2, 3]},
            cv=2,
        )
        grid.fit(X)
        assert grid.best_params_["n_clusters"] in [2, 3]

    def test_nan_euclidean_end_to_end(self):
        """Verify metric='nan_euclidean' works on missing data without ValueError."""
        rng = np.random.RandomState(42)
        X, _ = make_blobs(n_samples=40, centers=2, n_features=3, random_state=42)

        # Inject NaNs randomly (approx 5% of entries)
        mask = rng.rand(*X.shape) < 0.05
        X[mask] = np.nan

        model = CLARANS(n_clusters=2, metric="nan_euclidean", random_state=42)
        model.fit(X)

        assert len(model.medoid_indices_) == 2
        labels = model.predict(X)
        assert labels.shape == (40,)

        dists = model.transform(X)
        assert dists.shape == (40, 2)
        assert not np.isnan(dists).any()

        score = model.score(X)
        assert np.isfinite(score)

        # Tags check
        tags = model.__sklearn_tags__()
        assert tags.input_tags.allow_nan is True

    def test_precomputed_fortran_keeps_cython_active(self):
        """Verify precomputed F-contiguous distance matrix is handled cleanly."""
        D = np.array([
            [0.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
            [3.0, 1.0, 0.0]
        ], dtype=np.float64, order="F")

        model = CLARANS(n_clusters=2, metric="precomputed", random_state=42)
        model.fit(D)
        assert len(model.medoid_indices_) == 2
        assert model.labels_.shape == (3,)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython core not available")
    def test_fastpam1_delta_buffer_overflow_check(self):
        """Verify fastpam1_delta checks delta_buf length against n_clusters."""
        n_clusters = 5
        n_samples = 20
        delta_buf = np.zeros(3, dtype=np.float64)  # too small: 3 < 5
        d_xc = np.zeros(n_samples, dtype=np.float64)
        near_idx_map = np.zeros(n_samples, dtype=np.intp)
        near_dist = np.zeros(n_samples, dtype=np.float64)
        second_dist = np.zeros(n_samples, dtype=np.float64)

        with pytest.raises(ValueError, match="delta_buf length .* must be >= n_clusters"):
            fastpam1_delta(
                near_idx_map,
                near_dist,
                second_dist,
                d_xc,
                n_samples,
                n_clusters,
                delta_buf=delta_buf,
            )
