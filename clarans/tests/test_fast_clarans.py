import unittest

import numpy as np
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

        X_sparse = sparse.csr_matrix(self.X)
        model = FastCLARANS(n_clusters=3, numlocal=1, random_state=42)
        model.fit(X_sparse)
        labels = model.predict(X_sparse)
        self.assertEqual(labels.shape, (100,))

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


if __name__ == "__main__":
    unittest.main()
