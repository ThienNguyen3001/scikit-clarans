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
        model = FastCLARANS(n_clusters=3, numlocal=2, maxneighbor=10, random_state=42)
        model.fit(self.X)

        self.assertEqual(len(model.cluster_centers_), 3)
        self.assertEqual(len(model.labels_), 100)
        self.assertTrue(hasattr(model, "medoid_indices_"))

        for idx in model.medoid_indices_:
            self.assertTrue(np.any(np.all(self.X == self.X[idx], axis=1)))

    def test_predict(self):
        model = FastCLARANS(n_clusters=3, numlocal=1, random_state=42)
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
        model = FastCLARANS(n_clusters=3, numlocal=1, random_state=42)
        model.fit(X_sparse)
        labels = model.predict(X_sparse)
        self.assertEqual(labels.shape, (100,))

        # Test with csr_array if available
        if hasattr(sparse, "csr_array"):
            X_arr = sparse.csr_array(self.X)
            model_arr = FastCLARANS(n_clusters=3, numlocal=1, random_state=42)
            model_arr.fit(X_arr)
            labels_arr = model_arr.predict(X_arr)
            self.assertEqual(labels_arr.shape, (100,))

    def test_init_methods(self):
        for init_method in ["random", "heuristic", "k-medoids++", "build"]:
            model = FastCLARANS(
                n_clusters=3, numlocal=1, maxneighbor=10, init=init_method, random_state=42
            )
            model.fit(self.X)
            self.assertEqual(len(model.cluster_centers_), 3)

    def test_precomputed_init(self):
        init_centers = self.X[[0, 10, 20]]
        model = FastCLARANS(
            n_clusters=3, numlocal=1, maxneighbor=10, init=init_centers, random_state=42
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
        model = FastCLARANS(n_clusters=3, numlocal=1, maxneighbor=10, random_state=42)
        model.fit(self.X)
        X_transformed = model.transform(self.X)

        # Shape: (n_samples, n_clusters)
        self.assertEqual(X_transformed.shape, (100, 3))
        # All distances should be non-negative
        self.assertTrue(np.all(X_transformed >= 0))

    def test_cosine_metric(self):
        """Test that cosine metric works correctly."""
        model = FastCLARANS(
            n_clusters=3, numlocal=1, maxneighbor=10, metric="cosine", random_state=42
        )
        model.fit(self.X)
        self.assertEqual(len(model.cluster_centers_), 3)
        self.assertEqual(len(model.labels_), 100)
        labels = model.predict(self.X)
        np.testing.assert_array_equal(labels, model.labels_)

    def test_single_cluster(self):
        """Test with n_clusters=1 (edge case) and ensure swap optimizes cost."""
        model = FastCLARANS(n_clusters=1, numlocal=1, maxneighbor=10, random_state=42)
        model.fit(self.X)
        self.assertEqual(len(model.cluster_centers_), 1)
        self.assertTrue(np.all(model.labels_ == 0))

        # Test on 1D data where initial point is suboptimal to verify swap occurs
        X_1d = np.array([[0.0], [1.0], [10.0]])
        # random_state=0 initially selects index 2 (10.0) with cost 19.0
        model_1d = FastCLARANS(n_clusters=1, numlocal=1, maxneighbor=10, random_state=0)
        model_1d.fit(X_1d)
        self.assertEqual(model_1d.medoid_indices_[0], 1)
        self.assertAlmostEqual(model_1d.inertia_, 10.0)

    def test_inertia_attribute(self):
        """Test that inertia_ is set after fit and is non-negative."""
        model = FastCLARANS(n_clusters=3, numlocal=2, maxneighbor=50, random_state=42)
        model.fit(self.X)
        self.assertTrue(hasattr(model, "inertia_"))
        self.assertGreaterEqual(model.inertia_, 0)

    def test_n_iter_and_n_swaps_attributes(self):
        """Test that n_iter_ and n_swaps_ are set correctly in FastCLARANS."""
        model = FastCLARANS(n_clusters=3, numlocal=2, maxneighbor=50, random_state=42)
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
            FastCLARANS(numlocal=0).fit(self.X)
        with self.assertRaises(ValueError):
            FastCLARANS(maxneighbor=0).fit(self.X)
        with self.assertRaises(TypeError):
            FastCLARANS(max_iter=-1)

    def test_precomputed_metric(self):
        """Test FastCLARANS with metric='precomputed'."""
        from sklearn.metrics import pairwise_distances

        D = pairwise_distances(self.X, metric="euclidean")
        model = FastCLARANS(
            n_clusters=3, numlocal=2, maxneighbor=20, metric="precomputed", random_state=42
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
        """FastCLARANS should warn when numlocal > 1 with deterministic init and succeed."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = FastCLARANS(
                n_clusters=3, numlocal=2, maxneighbor=10, init="heuristic", random_state=42
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
        import pandas as pd

        model = FastCLARANS(n_clusters=3, random_state=42)
        model.set_output(transform="pandas")
        model.fit(self.X)
        transformed = model.transform(self.X)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertListEqual(
            list(transformed.columns),
            ["fastclarans0", "fastclarans1", "fastclarans2"],
        )


if __name__ == "__main__":
    unittest.main()
