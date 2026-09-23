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
"""

import io
import pickle
import unittest
import numpy as np

from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clarans import CLARANS, FastCLARANS, calculate_cost


class TestEcosystemIntegration(unittest.TestCase):
    """Group 1: Integration with Scikit-Learn Pipeline, Model Selection, and API standards."""

    def setUp(self):
        self.X, self.y = make_blobs(
            n_samples=120, centers=3, n_features=4, cluster_std=0.5, random_state=42
        )

    def test_pipeline_scaler_fit_predict(self):
        """Pipeline with StandardScaler followed by CLARANS / FastCLARANS."""
        for model_cls in [CLARANS, FastCLARANS]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clustering", model_cls(n_clusters=3, random_state=42)),
            ])
            pipe.fit(self.X)
            preds = pipe.predict(self.X)
            self.assertEqual(preds.shape, (len(self.X),))
            self.assertEqual(len(np.unique(preds)), 3)

    def test_pipeline_feature_extractor_with_classifier(self):
        """Using CLARANS / FastCLARANS transform() as distance features for LogisticRegression."""
        for model_cls in [CLARANS, FastCLARANS]:
            pipe = Pipeline([
                ("medoids", model_cls(n_clusters=3, random_state=42)),
                ("clf", LogisticRegression(random_state=42)),
            ])
            pipe.fit(self.X, self.y)
            score = pipe.score(self.X, self.y)
            self.assertGreater(score, 0.85)

    def test_grid_search_cv(self):
        """GridSearchCV tuning n_clusters on CLARANS via inertia scoring."""
        param_grid = {"n_clusters": [2, 3]}
        # Custom scoring function for clustering inertia
        cv = GridSearchCV(
            estimator=CLARANS(random_state=42),
            param_grid=param_grid,
            scoring=lambda est, X: -float(est.inertia_),
            cv=2,
        )
        cv.fit(self.X)
        self.assertIn(cv.best_params_["n_clusters"], [2, 3])
        self.assertTrue(hasattr(cv.best_estimator_, "medoid_indices_"))

    def test_estimator_clone(self):
        """Verify sklearn.base.clone creates an unfitted duplicate with identical params."""
        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=3, num_local=3, random_state=42)
            m.fit(self.X)

            m_cloned = clone(m)
            self.assertEqual(m.get_params(), m_cloned.get_params())
            self.assertFalse(hasattr(m_cloned, "medoid_indices_"))
            self.assertFalse(hasattr(m_cloned, "labels_"))
            self.assertFalse(hasattr(m_cloned, "inertia_"))

            m_cloned.fit(self.X)
            self.assertTrue(hasattr(m_cloned, "medoid_indices_"))
            np.testing.assert_array_equal(m.medoid_indices_, m_cloned.medoid_indices_)

    def test_fit_predict_consistency(self):
        """fit_predict(X) should yield exact same labels as fit(X).labels_."""
        for model_cls in [CLARANS, FastCLARANS]:
            m1 = model_cls(n_clusters=3, random_state=42)
            preds = m1.fit_predict(self.X)
            np.testing.assert_array_equal(preds, m1.labels_)

    def test_fit_transform_consistency(self):
        """fit_transform(X) should yield exact same matrix as fit(X).transform(X)."""
        for model_cls in [CLARANS, FastCLARANS]:
            m1 = model_cls(n_clusters=3, random_state=42)
            t1 = m1.fit_transform(self.X)
            t2 = m1.transform(self.X)
            np.testing.assert_allclose(t1, t2)


class TestSerialization(unittest.TestCase):
    """Group 2: Serialization and Deserialization via pickle and joblib."""

    def setUp(self):
        self.X, _ = make_blobs(n_samples=80, centers=3, n_features=2, random_state=42)
        self.X_new, _ = make_blobs(n_samples=20, centers=3, n_features=2, random_state=123)

    def test_pickle_roundtrip(self):
        """pickle dumps & loads preserves all model states and prediction outputs."""
        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=3, random_state=42).fit(self.X)
            serialized = pickle.dumps(m)
            m_loaded = pickle.loads(serialized)

            np.testing.assert_array_equal(m.medoid_indices_, m_loaded.medoid_indices_)
            self.assertAlmostEqual(m.inertia_, m_loaded.inertia_, places=9)
            np.testing.assert_array_equal(m.predict(self.X_new), m_loaded.predict(self.X_new))
            np.testing.assert_allclose(m.transform(self.X_new), m_loaded.transform(self.X_new))

    def test_joblib_roundtrip(self):
        """joblib dump & load roundtrip."""
        try:
            import joblib
        except ImportError:
            self.skipTest("joblib is not installed")

        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=3, random_state=42).fit(self.X)
            buf = io.BytesIO()
            joblib.dump(m, buf)
            buf.seek(0)
            m_loaded = joblib.load(buf)

            np.testing.assert_array_equal(m.medoid_indices_, m_loaded.medoid_indices_)
            self.assertAlmostEqual(m.inertia_, m_loaded.inertia_, places=9)
            np.testing.assert_array_equal(m.predict(self.X_new), m_loaded.predict(self.X_new))


class TestMemoryAndDtypes(unittest.TestCase):
    """Group 4: Non-standard dtypes and memory layouts."""

    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)

    def test_float32_input(self):
        """float32 input should execute smoothly across Cython kernels."""
        X_f32 = self.X.astype(np.float32)
        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=3, random_state=42).fit(X_f32)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertEqual(m.labels_.shape, (100,))
            preds = m.predict(X_f32)
            self.assertEqual(preds.shape, (100,))

    def test_fortran_contiguous_input(self):
        """Fortran-ordered (column-major) input should be accepted without memory errors."""
        X_fortran = np.asfortranarray(self.X)
        self.assertTrue(X_fortran.flags.f_contiguous)
        self.assertFalse(X_fortran.flags.c_contiguous)

        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=3, random_state=42).fit(X_fortran)
            self.assertEqual(len(m.medoid_indices_), 3)
            self.assertEqual(m.labels_.shape, (100,))

    def test_strided_slice_input(self):
        """Non-contiguous strided slice input."""
        X_strided = self.X[::2, :]
        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=2, random_state=42).fit(X_strided)
            self.assertEqual(len(m.medoid_indices_), 2)
            self.assertEqual(m.labels_.shape, (50,))

    def test_sparse_csc_matrix(self):
        """CSC sparse matrix input."""
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("scipy is not installed")

        X_csc = sparse.csc_matrix(self.X)
        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=3, random_state=42).fit(X_csc)
            self.assertEqual(len(m.medoid_indices_), 3)
            preds = m.predict(X_csc)
            self.assertEqual(preds.shape, (100,))


class TestMathematicalInvariance(unittest.TestCase):
    """Group 5: Mathematical equivalence and cluster agreement."""

    def test_delta_vs_brute_force_exact_match(self):
        """CLARANS cost_evaluation='delta' vs 'brute_force' must yield
        bit-identical medoids and inertia."""
        X, _ = make_blobs(n_samples=80, centers=3, n_features=2, random_state=42)
        m_delta = CLARANS(n_clusters=3, cost_evaluation="delta", random_state=42).fit(X)
        m_bf = CLARANS(n_clusters=3, cost_evaluation="brute_force", random_state=42).fit(X)

        np.testing.assert_array_equal(m_delta.medoid_indices_, m_bf.medoid_indices_)
        self.assertAlmostEqual(m_delta.inertia_, m_bf.inertia_, places=9)
        np.testing.assert_array_equal(m_delta.labels_, m_bf.labels_)

    def test_clarans_fast_clarans_cluster_agreement(self):
        """CLARANS and FastCLARANS should find high-quality matching clusters
        on well-separated blobs."""
        X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.3, random_state=42)
        m_clarans = CLARANS(n_clusters=3, random_state=42).fit(X)
        m_fast = FastCLARANS(n_clusters=3, random_state=42).fit(X)

        ari = adjusted_rand_score(m_clarans.labels_, m_fast.labels_)
        self.assertGreaterEqual(ari, 0.90)


class TestBoundaryAndCornerCases(unittest.TestCase):
    """Group 6: Edge cases, minimal inputs, and duplicate points."""

    def test_all_duplicate_points(self):
        """Input with duplicate rows (e.g. 60 samples with only 3 distinct coordinate values)."""
        unique_pts = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        X_dup = np.repeat(unique_pts, 20, axis=0)

        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=3, random_state=42).fit(X_dup)
            self.assertEqual(len(m.medoid_indices_), 3)
            # Distinct medoid coordinates should cover the 3 unique points
            chosen_pts = np.unique(X_dup[m.medoid_indices_], axis=0)
            self.assertEqual(len(chosen_pts), 3)

    def test_n_clusters_greater_or_equal_n_samples_raises(self):
        """n_clusters >= n_samples must raise ValueError."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        for model_cls in [CLARANS, FastCLARANS]:
            # n_clusters == n_samples
            with self.assertRaises(ValueError):
                model_cls(n_clusters=3).fit(X)
            # n_clusters > n_samples
            with self.assertRaises(ValueError):
                model_cls(n_clusters=4).fit(X)

    def test_minimal_dataset(self):
        """Minimal allowable datasets: N=2, K=1 and N=3, K=2."""
        X_n2 = np.array([[1.0, 2.0], [5.0, 6.0]])
        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=1, random_state=42).fit(X_n2)
            self.assertEqual(len(m.medoid_indices_), 1)
            self.assertEqual(len(m.labels_), 2)

        X_n3 = np.array([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=2, random_state=42).fit(X_n3)
            self.assertEqual(len(m.medoid_indices_), 2)
            self.assertEqual(len(m.labels_), 3)


class TestCustomCallableMetric(unittest.TestCase):
    """Group 7: Custom callable distance metric end-to-end."""

    def test_custom_metric_pipeline(self):
        """Verify custom Python distance callable functions across fit, predict,
        transform, and calculate_cost."""
        def custom_cityblock(u, v):
            return float(np.sum(np.abs(u - v)))

        X = np.array([[0.0, 0.0], [0.5, 0.5], [10.0, 10.0], [10.5, 10.5]])

        for model_cls in [CLARANS, FastCLARANS]:
            m = model_cls(n_clusters=2, metric=custom_cityblock, random_state=42)
            m.fit(X)

            self.assertEqual(len(m.medoid_indices_), 2)
            preds = m.predict(X)
            self.assertEqual(preds.shape, (4,))

            dists = m.transform(X)
            self.assertEqual(dists.shape, (4, 2))

            cost = calculate_cost(X, m.medoid_indices_, custom_cityblock)
            self.assertAlmostEqual(cost, m.inertia_, places=9)


if __name__ == "__main__":
    unittest.main()
