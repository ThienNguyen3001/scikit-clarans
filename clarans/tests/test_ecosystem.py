"""
Comprehensive Advanced Test Suite for CLARANS & FastCLARANS.

Covers:
- Group 1: Scikit-Learn Ecosystem Integration (Pipeline, GridSearchCV, clone,
  fit_predict, fit_transform)
- Group 2: Model Serialization (Pickle & Joblib roundtrips)
- Group 4: Data Types & Memory Layout (float32, Fortran order, strided slices, CSC sparse)
- Group 5: Mathematical Invariance (delta vs brute_force equivalence,
  ARI between CLARANS and FastCLARANS)
- Group 6: Boundary & Corner Cases (Duplicate points, n_clusters >= n_samples ValueError,
  minimal N=2, K=1)
- Group 7: Custom Callable Distance Metric End-to-End
- Group 8: Parallelism and Thread Safety across Local Restarts (n_jobs)
"""

import io
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clarans import CLARANS, FastCLARANS, calculate_cost


class TestEcosystemIntegration:
    """Group 1: Integration with Scikit-Learn Pipeline, Model Selection, and API standards."""

    def setup_method(self):
        self.X, self.y = make_blobs(
            n_samples=120, centers=3, n_features=4, cluster_std=0.5, random_state=42
        )

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_pipeline_scaler_fit_predict(self, model_cls):
        """Pipeline with StandardScaler followed by CLARANS / FastCLARANS."""
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clustering", model_cls(n_clusters=3, random_state=42)),
            ]
        )
        pipe.fit(self.X)
        preds = pipe.predict(self.X)
        assert preds.shape == (len(self.X),)
        assert len(np.unique(preds)) == 3

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_pipeline_feature_extractor_with_classifier(self, model_cls):
        """Using CLARANS / FastCLARANS transform() as distance features for LogisticRegression."""
        pipe = Pipeline(
            [
                ("medoids", model_cls(n_clusters=3, random_state=42)),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )
        pipe.fit(self.X, self.y)
        score = pipe.score(self.X, self.y)
        assert score > 0.85

    def test_grid_search_cv(self):
        """GridSearchCV tuning n_clusters on CLARANS via inertia scoring."""
        param_grid = {"n_clusters": [2, 3]}
        cv = GridSearchCV(
            estimator=CLARANS(random_state=42),
            param_grid=param_grid,
            scoring=lambda est, X: -float(est.inertia_),
            cv=2,
        )
        cv.fit(self.X)
        assert cv.best_params_["n_clusters"] in [2, 3]
        assert hasattr(cv.best_estimator_, "medoid_indices_")

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_estimator_clone(self, model_cls):
        """Verify sklearn.base.clone creates an unfitted duplicate with identical params."""
        m = model_cls(n_clusters=3, num_local=3, random_state=42)
        m.fit(self.X)

        m_cloned = clone(m)
        assert m.get_params() == m_cloned.get_params()
        assert not hasattr(m_cloned, "medoid_indices_")
        assert not hasattr(m_cloned, "labels_")
        assert not hasattr(m_cloned, "inertia_")

        m_cloned.fit(self.X)
        assert hasattr(m_cloned, "medoid_indices_")
        np.testing.assert_array_equal(m.medoid_indices_, m_cloned.medoid_indices_)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_fit_predict_consistency(self, model_cls):
        """fit_predict(X) should yield exact same labels as fit(X).labels_."""
        m1 = model_cls(n_clusters=3, random_state=42)
        preds = m1.fit_predict(self.X)
        np.testing.assert_array_equal(preds, m1.labels_)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_fit_transform_consistency(self, model_cls):
        """fit_transform(X) should yield exact same matrix as fit(X).transform(X)."""
        m1 = model_cls(n_clusters=3, random_state=42)
        t1 = m1.fit_transform(self.X)
        t2 = m1.transform(self.X)
        np.testing.assert_allclose(t1, t2)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_pipeline_and_gridsearch_precomputed_n_features_in(self, model_cls):
        """Pipeline and GridSearchCV integration with metric='precomputed' sets n_features_in_."""
        D = pairwise_distances(self.X[:30])
        pipe = Pipeline(
            [("clustering", model_cls(n_clusters=2, metric="precomputed", random_state=42))]
        )
        pipe.fit(D)
        assert pipe.n_features_in_ == 30

        grid = GridSearchCV(
            model_cls(metric="precomputed", random_state=42),
            param_grid={"n_clusters": [2, 3]},
            cv=2,
        )
        grid.fit(D)
        assert grid.n_features_in_ == 30


class TestSerialization:
    """Group 2: Serialization and Deserialization via pickle and joblib."""

    def setup_method(self):
        self.X, _ = make_blobs(n_samples=80, centers=3, n_features=2, random_state=42)
        self.X_new, _ = make_blobs(n_samples=20, centers=3, n_features=2, random_state=123)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_pickle_roundtrip(self, model_cls):
        """pickle dumps & loads preserves all model states and prediction outputs."""
        m = model_cls(n_clusters=3, random_state=42).fit(self.X)
        serialized = pickle.dumps(m)
        m_loaded = pickle.loads(serialized)

        np.testing.assert_array_equal(m.medoid_indices_, m_loaded.medoid_indices_)
        assert m.inertia_ == pytest.approx(m_loaded.inertia_, abs=1e-9)
        np.testing.assert_array_equal(m.predict(self.X_new), m_loaded.predict(self.X_new))
        np.testing.assert_allclose(m.transform(self.X_new), m_loaded.transform(self.X_new))

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_joblib_roundtrip(self, model_cls):
        """joblib dump & load roundtrip."""
        joblib = pytest.importorskip("joblib")

        m = model_cls(n_clusters=3, random_state=42).fit(self.X)
        buf = io.BytesIO()
        joblib.dump(m, buf)
        buf.seek(0)
        m_loaded = joblib.load(buf)

        np.testing.assert_array_equal(m.medoid_indices_, m_loaded.medoid_indices_)
        assert m.inertia_ == pytest.approx(m_loaded.inertia_, abs=1e-9)
        np.testing.assert_array_equal(m.predict(self.X_new), m_loaded.predict(self.X_new))


class TestMemoryAndDtypes:
    """Group 4: Non-standard dtypes and memory layouts."""

    def setup_method(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_float32_input(self, model_cls):
        """float32 input should execute smoothly across Cython kernels."""
        X_f32 = self.X.astype(np.float32)
        m = model_cls(n_clusters=3, random_state=42).fit(X_f32)
        assert len(m.medoid_indices_) == 3
        assert m.labels_.shape == (100,)
        preds = m.predict(X_f32)
        assert preds.shape == (100,)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_fortran_contiguous_input(self, model_cls):
        """Fortran-ordered (column-major) input should be accepted without memory errors."""
        X_fortran = np.asfortranarray(self.X)
        assert X_fortran.flags.f_contiguous
        assert not X_fortran.flags.c_contiguous

        m = model_cls(n_clusters=3, random_state=42).fit(X_fortran)
        assert len(m.medoid_indices_) == 3
        assert m.labels_.shape == (100,)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_strided_slice_input(self, model_cls):
        """Non-contiguous strided slice input."""
        X_strided = self.X[::2, :]
        m = model_cls(n_clusters=2, random_state=42).fit(X_strided)
        assert len(m.medoid_indices_) == 2
        assert m.labels_.shape == (50,)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_sparse_csc_matrix(self, model_cls):
        """CSC sparse matrix input."""
        sparse = pytest.importorskip("scipy.sparse")

        X_csc = sparse.csc_matrix(self.X)
        m = model_cls(n_clusters=3, random_state=42).fit(X_csc)
        assert len(m.medoid_indices_) == 3
        preds = m.predict(X_csc)
        assert preds.shape == (100,)


class TestMathematicalInvariance:
    """Group 5: Mathematical equivalence and cluster agreement."""

    def test_delta_vs_brute_force_exact_match(self):
        """CLARANS cost_evaluation='delta' vs 'brute_force' must yield
        bit-identical medoids and inertia."""
        X, _ = make_blobs(n_samples=80, centers=3, n_features=2, random_state=42)
        m_delta = CLARANS(n_clusters=3, cost_evaluation="delta", random_state=42).fit(X)
        m_bf = CLARANS(n_clusters=3, cost_evaluation="brute_force", random_state=42).fit(X)

        np.testing.assert_array_equal(m_delta.medoid_indices_, m_bf.medoid_indices_)
        assert m_delta.inertia_ == pytest.approx(m_bf.inertia_, abs=1e-9)
        np.testing.assert_array_equal(m_delta.labels_, m_bf.labels_)

    def test_clarans_fast_clarans_cluster_agreement(self):
        """CLARANS and FastCLARANS should find high-quality matching clusters
        on well-separated blobs."""
        X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.3, random_state=42)
        m_clarans = CLARANS(n_clusters=3, random_state=42).fit(X)
        m_fast = FastCLARANS(n_clusters=3, random_state=42).fit(X)

        ari = adjusted_rand_score(m_clarans.labels_, m_fast.labels_)
        assert ari >= 0.90


class TestBoundaryAndCornerCases:
    """Group 6: Edge cases, minimal inputs, and duplicate points."""

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_all_duplicate_points(self, model_cls):
        """Input with duplicate rows (e.g. 60 samples with only 3 distinct coordinate values)."""
        unique_pts = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        X_dup = np.repeat(unique_pts, 20, axis=0)

        m = model_cls(n_clusters=3, random_state=42).fit(X_dup)
        assert len(m.medoid_indices_) == 3
        # Distinct medoid coordinates should cover the 3 unique points
        chosen_pts = np.unique(X_dup[m.medoid_indices_], axis=0)
        assert len(chosen_pts) == 3

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_n_clusters_greater_or_equal_n_samples_raises(self, model_cls):
        """n_clusters >= n_samples must raise ValueError."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # n_clusters == n_samples
        with pytest.raises(ValueError):
            model_cls(n_clusters=3).fit(X)
        # n_clusters > n_samples
        with pytest.raises(ValueError):
            model_cls(n_clusters=4).fit(X)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_minimal_dataset(self, model_cls):
        """Minimal allowable datasets: N=2, K=1 and N=3, K=2."""
        X_n2 = np.array([[1.0, 2.0], [5.0, 6.0]])
        m = model_cls(n_clusters=1, random_state=42).fit(X_n2)
        assert len(m.medoid_indices_) == 1
        assert len(m.labels_) == 2

        X_n3 = np.array([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
        m = model_cls(n_clusters=2, random_state=42).fit(X_n3)
        assert len(m.medoid_indices_) == 2
        assert len(m.labels_) == 3


class TestCustomCallableMetric:
    """Group 7: Custom callable distance metric end-to-end."""

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_custom_metric_pipeline(self, model_cls):
        """Verify custom Python distance callable functions across fit, predict,
        transform, and calculate_cost."""

        def custom_cityblock(u, v):
            return float(np.sum(np.abs(u - v)))

        X = np.array([[0.0, 0.0], [0.5, 0.5], [10.0, 10.0], [10.5, 10.5]])

        m = model_cls(n_clusters=2, metric=custom_cityblock, random_state=42)
        m.fit(X)

        assert len(m.medoid_indices_) == 2
        preds = m.predict(X)
        assert preds.shape == (4,)

        dists = m.transform(X)
        assert dists.shape == (4, 2)

        cost = calculate_cost(X, m.medoid_indices_, custom_cityblock)
        assert cost == pytest.approx(m.inertia_, abs=1e-9)


class TestNJobsParallelism:
    """Group 8: Verification of n_jobs parameter in fit restarts."""

    def setup_method(self):
        self.X, _ = make_blobs(
            n_samples=200, centers=4, n_features=6, cluster_std=0.8, random_state=42
        )
        self.X_test, _ = make_blobs(
            n_samples=50, centers=4, n_features=6, cluster_std=0.8, random_state=99
        )

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_n_jobs_fit_equivalence(self, model_cls):
        """Ensure n_jobs=None, 1, 2, and -1 give 100% bit-for-bit identical results."""
        base = model_cls(n_clusters=4, num_local=3, random_state=42, n_jobs=None).fit(self.X)
        base_pred = base.predict(self.X_test)
        base_trans = base.transform(self.X_test)
        base_score = base.score(self.X_test)

        for nj in [1, 2, -1]:
            m = model_cls(n_clusters=4, num_local=3, random_state=42, n_jobs=nj).fit(self.X)
            # Centers, medoids, inertia and labels must match bit-for-bit
            np.testing.assert_array_equal(m.medoid_indices_, base.medoid_indices_)
            np.testing.assert_allclose(m.cluster_centers_, base.cluster_centers_)
            assert m.inertia_ == pytest.approx(base.inertia_, abs=1e-9)
            np.testing.assert_array_equal(m.labels_, base.labels_)

            # Predict, transform, score must match bit-for-bit
            np.testing.assert_array_equal(m.predict(self.X_test), base_pred)
            np.testing.assert_allclose(m.transform(self.X_test), base_trans, atol=1e-12)
            assert m.score(self.X_test) == pytest.approx(base_score, abs=1e-9)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_n_jobs_validation(self, model_cls):
        """Verify invalid n_jobs values raise ValueError."""
        for invalid in [0, True, False, 1.5, "all"]:
            with pytest.raises(ValueError):
                model_cls(n_jobs=invalid).fit(self.X)

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_n_jobs_clone_and_get_params(self, model_cls):
        """Verify clone and get_params properly preserve n_jobs."""
        m = model_cls(n_clusters=3, n_jobs=4, random_state=42)
        assert m.get_params()["n_jobs"] == 4
        cloned = clone(m)
        assert cloned.n_jobs == 4
        cloned.set_params(n_jobs=-1)
        assert cloned.n_jobs == -1

    @pytest.mark.parametrize("model_cls", [CLARANS, FastCLARANS])
    def test_n_jobs_nested_parallelism_grid_search_cv(self, model_cls):
        """Verify nested parallelism when GridSearchCV(n_jobs=2) wraps CLARANS(n_jobs=2)."""
        param_grid = {"n_clusters": [2, 3]}
        grid = GridSearchCV(
            estimator=model_cls(num_local=2, random_state=42, n_jobs=2),
            param_grid=param_grid,
            cv=2,
            n_jobs=2,
        )
        grid.fit(self.X)
        assert grid.best_params_["n_clusters"] in [2, 3]
        assert hasattr(grid.best_estimator_, "medoid_indices_")
