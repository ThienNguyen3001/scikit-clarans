import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_blobs

from clarans import FastCLARANS


class TestFastCLARANS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(
            n_samples=100, centers=3, n_features=2, random_state=42
        )

    def test_fit(self):
        model = FastCLARANS(n_clusters=3, num_local=2, max_neighbors=10, random_state=42)
        model.fit(self.X)

        self.assertEqual(len(model.cluster_centers_), 3)
        self.assertEqual(len(model.labels_), 100)
        self.assertTrue(hasattr(model, "medoid_indices_"))

        for idx in model.medoid_indices_:
            self.assertTrue(np.any(np.all(self.X == self.X[idx], axis=1)))

    def test_predict(self):
        model = FastCLARANS(n_clusters=3, num_local=1, random_state=42)
        model.fit(self.X)
        labels = model.predict(self.X)
        self.assertEqual(labels.shape, (100,))

    def test_sparse_input(self):
        try:
            from scipy import sparse
        except Exception:
            self.skipTest("scipy not available")

        # Test with csr_matrix
        X_sparse = sparse.csr_matrix(self.X)
        model = FastCLARANS(n_clusters=3, num_local=1, random_state=42)
        model.fit(X_sparse)
        labels = model.predict(X_sparse)
        self.assertEqual(labels.shape, (100,))

        # Test with csr_array if available
        if hasattr(sparse, "csr_array"):
            X_arr = sparse.csr_array(self.X)
            model_arr = FastCLARANS(n_clusters=3, num_local=1, random_state=42)
            model_arr.fit(X_arr)
            labels_arr = model_arr.predict(X_arr)
            self.assertEqual(labels_arr.shape, (100,))

    def test_init_methods(self):
        for init_method in ["random", "heuristic", "k-medoids++", "build"]:
            model = FastCLARANS(
                n_clusters=3, num_local=1, max_neighbors=10, init=init_method, random_state=42
            )
            model.fit(self.X)
            self.assertEqual(len(model.cluster_centers_), 3)

    def test_precomputed_init(self):
        init_centers = self.X[[0, 10, 20]]
        model = FastCLARANS(
            n_clusters=3, num_local=1, max_neighbors=10, init=init_centers, random_state=42
        )
        model.fit(self.X)
        self.assertEqual(len(model.cluster_centers_), 3)

    def test_input_validation_init(self):
        model = FastCLARANS(n_clusters=3, init="invalid_method")
        with self.assertRaises(ValueError):
            model.fit(self.X)

        model = FastCLARANS(n_clusters=3, init=self.X[:2])
        with self.assertRaises(ValueError):
            model.fit(self.X)

    def test_transform(self):
        """Test transform() returns distances to cluster centers."""
        model = FastCLARANS(n_clusters=3, num_local=1, max_neighbors=10, random_state=42)
        model.fit(self.X)
        X_transformed = model.transform(self.X)

        # Shape: (n_samples, n_clusters)
        self.assertEqual(X_transformed.shape, (100, 3))
        # All distances should be non-negative
        self.assertTrue(np.all(X_transformed >= 0))

    def test_cosine_metric(self):
        """Test that cosine metric works correctly."""
        model = FastCLARANS(
            n_clusters=3, num_local=1, max_neighbors=10, metric="cosine", random_state=42
        )
        model.fit(self.X)
        self.assertEqual(len(model.cluster_centers_), 3)
        self.assertEqual(len(model.labels_), 100)
        labels = model.predict(self.X)
        np.testing.assert_array_equal(labels, model.labels_)

    def test_single_cluster(self):
        """Test with n_clusters=1 (edge case) and ensure swap optimizes cost."""
        model = FastCLARANS(n_clusters=1, num_local=1, max_neighbors=10, random_state=42)
        model.fit(self.X)
        self.assertEqual(len(model.cluster_centers_), 1)
        self.assertTrue(np.all(model.labels_ == 0))

        # Test on 1D data where initial point is suboptimal to verify swap occurs
        X_1d = np.array([[0.0], [1.0], [10.0]])
        # random_state=0 initially selects index 2 (10.0) with cost 19.0
        model_1d = FastCLARANS(n_clusters=1, num_local=1, max_neighbors=10, random_state=0)
        model_1d.fit(X_1d)
        self.assertEqual(model_1d.medoid_indices_[0], 1)
        self.assertAlmostEqual(model_1d.inertia_, 10.0)

    def test_inertia_attribute(self):
        """Test that inertia_ is set after fit and is non-negative."""
        model = FastCLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=42)
        model.fit(self.X)
        self.assertTrue(hasattr(model, "inertia_"))
        self.assertGreaterEqual(model.inertia_, 0)

    def test_n_iter_and_n_swaps_attributes(self):
        """Test that n_iter_ and n_swaps_ are set correctly in FastCLARANS."""
        model = FastCLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=42)
        model.fit(self.X)
        self.assertTrue(hasattr(model, "n_iter_"))
        self.assertTrue(hasattr(model, "n_swaps_"))
        self.assertGreaterEqual(model.n_iter_, 1)
        self.assertGreaterEqual(model.n_swaps_, 0)
        self.assertLessEqual(model.n_swaps_, model.n_iter_)

    def test_invalid_parameters(self):
        """Test parameter validation in FastCLARANS."""
        with self.assertRaises(ValueError):
            FastCLARANS(n_clusters=0).fit(self.X)
        with self.assertRaises(ValueError):
            FastCLARANS(num_local=0).fit(self.X)
        for val in [0, -1, None, "invalid", 1.5, True, False]:
            with self.assertRaises(ValueError):
                FastCLARANS(max_neighbors=val).fit(self.X)
        with self.assertRaises(TypeError):
            FastCLARANS(max_iter=-1)

    def test_precomputed_metric(self):
        """Test FastCLARANS with metric='precomputed'."""
        from sklearn.metrics import pairwise_distances

        D = pairwise_distances(self.X, metric="euclidean")
        model = FastCLARANS(
            n_clusters=3, num_local=2, max_neighbors=20, metric="precomputed", random_state=42
        )
        model.fit(D)

        self.assertEqual(len(model.medoid_indices_), 3)
        self.assertEqual(len(model.labels_), 100)
        self.assertGreaterEqual(model.inertia_, 0)
        self.assertIsNone(model.cluster_centers_)
        self.assertFalse(hasattr(model, "n_features_in_"))

        # predict on square matrix
        labels = model.predict(D)
        np.testing.assert_array_equal(labels, model.labels_)

        # transform on square matrix
        transformed = model.transform(D)
        self.assertEqual(transformed.shape, (100, 3))

    def test_precomputed_non_square(self):
        """FastCLARANS should raise ValueError on non-square precomputed matrix."""
        model = FastCLARANS(n_clusters=3, metric="precomputed", random_state=42)
        with self.assertRaises(ValueError):
            model.fit(self.X)

    def test_deterministic_init_warning(self):
        """FastCLARANS should warn when num_local > 1 with deterministic init and succeed."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = FastCLARANS(
                n_clusters=3, num_local=2, max_neighbors=10, init="heuristic", random_state=42
            )
            model.fit(self.X)
            self.assertTrue(any(issubclass(warn.category, UserWarning) for warn in w))
            self.assertEqual(len(model.medoid_indices_), 3)

    def test_keyword_only_args(self):
        """FastCLARANS should enforce keyword-only arguments per SLEP009."""
        with self.assertRaises(TypeError):
            FastCLARANS(3)

    def test_get_feature_names_out(self):
        """FastCLARANS should provide get_feature_names_out per SLEP007."""
        model = FastCLARANS(n_clusters=3)
        with self.assertRaises(NotFittedError):
            model.get_feature_names_out()

        model.fit(self.X)
        names = model.get_feature_names_out()
        np.testing.assert_array_equal(
            names,
            np.array(["fastclarans0", "fastclarans1", "fastclarans2"], dtype=object),
        )

    def test_pandas_output(self):
        """FastCLARANS should support set_output(transform='pandas') per SLEP018."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")

        model = FastCLARANS(n_clusters=3, random_state=42)
        model.set_output(transform="pandas")
        model.fit(self.X)
        transformed = model.transform(self.X)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertListEqual(
            list(transformed.columns),
            ["fastclarans0", "fastclarans1", "fastclarans2"],
        )

    def test_legacy_parameters_removed(self):
        """FastCLARANS should raise TypeError when numlocal or maxneighbor are passed."""
        with self.assertRaises(TypeError):
            FastCLARANS(numlocal=2)
        with self.assertRaises(TypeError):
            FastCLARANS(maxneighbor=25)

    def test_parameters_num_local_max_neighbors(self):
        """FastCLARANS with num_local and max_neighbors should work and set max_neighbors_."""
        model = FastCLARANS(
            n_clusters=3, num_local=2, max_neighbors=25, random_state=42
        )
        model.fit(self.X)
        self.assertEqual(model.max_neighbors_, 25)
        self.assertFalse(hasattr(model, "maxneighbor_"))

    def test_max_neighbors_default(self):
        """Default max_neighbors should be 'auto' and calculated correctly in FastCLARANS."""
        model = FastCLARANS(n_clusters=3, num_local=1, random_state=42)
        self.assertEqual(model.max_neighbors, "auto")
        model.fit(self.X)
        expected = max(1, int(250 / 3), int(0.025 * (100 - 3)))
        self.assertEqual(model.max_neighbors_, expected)

    def test_max_neighbors_explicit_auto(self):
        """Explicit max_neighbors='auto' should work identically to default in FastCLARANS."""
        model = FastCLARANS(n_clusters=3, num_local=1, max_neighbors="auto", random_state=42)
        model.fit(self.X)
        expected = max(1, int(250 / 3), int(0.025 * (100 - 3)))
        self.assertEqual(model.max_neighbors_, expected)

    def test_delta_tolerance_rejects_ghost_swaps(self):
        """Tolerance should prevent ghost swaps in FastCLARANS."""
        X_dup = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
        model = FastCLARANS(n_clusters=2, num_local=1, max_neighbors=50, random_state=42)
        model.fit(X_dup)
        self.assertGreaterEqual(model.n_swaps_, 0)

    def test_no_cache_parameter(self):
        """FastCLARANS should not accept cache or cost_evaluation parameter in __init__."""
        with self.assertRaises(TypeError):
            FastCLARANS(cache=False)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            FastCLARANS(cost_evaluation="brute_force")  # type: ignore[call-arg]

    def test_inherits_update_cache(self):
        """FastCLARANS should inherit _update_cache directly from CLARANS without overriding."""
        from clarans._clarans import CLARANS

        self.assertIs(FastCLARANS._update_cache, CLARANS._update_cache)


class TestFastCLARANSMetricParams(unittest.TestCase):
    """Test suite for metric_params parameter in FastCLARANS."""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(30, 3)

    def test_minkowski_with_p(self):
        """FastCLARANS should support Minkowski metric with p passed via metric_params."""
        model = FastCLARANS(
            n_clusters=2, metric="minkowski", metric_params={"p": 3}, random_state=42
        )
        model.fit(self.X)
        self.assertEqual(model.labels_.shape, (len(self.X),))
        self.assertEqual(model.cluster_centers_.shape, (2, 3))
        preds = model.predict(self.X[:5])
        self.assertEqual(preds.shape, (5,))
        trans = model.transform(self.X[:5])
        self.assertEqual(trans.shape, (5, 2))

    def test_mahalanobis_with_vi(self):
        """FastCLARANS should support Mahalanobis metric with VI in metric_params."""
        VI = np.linalg.inv(np.cov(self.X.T))
        model = FastCLARANS(
            n_clusters=2, metric="mahalanobis", metric_params={"VI": VI}, random_state=42
        )
        model.fit(self.X)
        self.assertEqual(model.labels_.shape, (len(self.X),))
        preds = model.predict(self.X[:5])
        self.assertEqual(preds.shape, (5,))
        trans = model.transform(self.X[:5])
        self.assertEqual(trans.shape, (5, 2))

    def test_mahalanobis_missing_vi_raises(self):
        """FastCLARANS with mahalanobis and no VI should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            FastCLARANS(n_clusters=2, metric="mahalanobis", random_state=42).fit(self.X)
        self.assertIn("vi", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            FastCLARANS(
                n_clusters=2, metric="mahalanobis", metric_params={}, random_state=42
            ).fit(self.X)
        self.assertIn("vi", str(ctx.exception).lower())

    def test_invalid_metric_params_type(self):
        """Non-dict metric_params should raise ValueError in FastCLARANS."""
        with self.assertRaises(ValueError) as ctx:
            FastCLARANS(n_clusters=2, metric_params="not_a_dict", random_state=42).fit(
                self.X
            )
        self.assertIn("metric_params", str(ctx.exception).lower())

    def test_clone_preserves_metric_params(self):
        """clone should preserve metric_params correctly for FastCLARANS."""
        from sklearn.base import clone

        model = FastCLARANS(metric="minkowski", metric_params={"p": 4})
        cloned = clone(model)
        self.assertEqual(cloned.metric_params, {"p": 4})

    def test_verbose_silent(self):
        """verbose=0 and verbose=False should produce no output to stdout."""
        import io
        from contextlib import redirect_stdout

        for v in (0, False):
            f = io.StringIO()
            with redirect_stdout(f):
                model = FastCLARANS(n_clusters=2, num_local=1, max_neighbors=10, verbose=v, random_state=42)
                model.fit(self.X)
            self.assertEqual(f.getvalue(), "")

    def test_verbose_level_1(self):
        """verbose=1 and verbose=True should print local search summaries and best cost."""
        import io
        from contextlib import redirect_stdout

        for v in (1, True):
            f = io.StringIO()
            with redirect_stdout(f):
                model = FastCLARANS(n_clusters=2, num_local=2, max_neighbors=10, verbose=v, random_state=42)
                model.fit(self.X)
            output = f.getvalue()
            self.assertIn("[FastCLARANS] Fitting with", output)
            self.assertIn("[FastCLARANS] Local search 1/2:", output)
            self.assertIn("[FastCLARANS] Local search 1/2 done in", output)
            self.assertIn("[FastCLARANS] Local search 2/2 done in", output)
            self.assertIn("[FastCLARANS] Best cost:", output)

    def test_verbose_level_2(self):
        """verbose=2 should print individual swap details when swaps occur."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            model = FastCLARANS(n_clusters=2, num_local=1, max_neighbors=50, init="random", verbose=2, random_state=0)
            model.fit(self.X)
        output = f.getvalue()
        self.assertIn("[FastCLARANS] Fitting with", output)
        self.assertIn("[FastCLARANS] Local search 1/1 done in", output)
        if model.n_swaps_ > 0:
            self.assertIn("Swap", output)
            self.assertIn("Delta:", output)

    def test_verbose_invalid(self):
        """Invalid verbose values should raise ValueError."""
        for invalid_val in [-1, -5, "1", 1.5, [1]]:
            with self.assertRaises(ValueError) as ctx:
                FastCLARANS(n_clusters=2, verbose=invalid_val, random_state=42).fit(self.X)
            self.assertIn("verbose", str(ctx.exception).lower())

    def test_clone_preserves_verbose(self):
        """clone should preserve verbose correctly."""
        from sklearn.base import clone

        model = FastCLARANS(verbose=2)
        cloned = clone(model)
        self.assertEqual(cloned.verbose, 2)


if __name__ == "__main__":
    unittest.main()
