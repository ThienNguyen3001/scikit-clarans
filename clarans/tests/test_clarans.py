import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from clarans import CLARANS


class TestCLARANS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(
            n_samples=100, centers=3, n_features=2, random_state=42
        )

    def test_fit(self):
        clarans = CLARANS(n_clusters=3, numlocal=2, maxneighbor=10, random_state=42)
        clarans.fit(self.X)

        self.assertEqual(len(clarans.cluster_centers_), 3)
        self.assertEqual(len(clarans.labels_), 100)
        self.assertTrue(hasattr(clarans, "medoid_indices_"))

        for idx in clarans.medoid_indices_:
            self.assertTrue(np.any(np.all(self.X == self.X[idx], axis=1)))

    def test_predict(self):
        clarans = CLARANS(n_clusters=3, numlocal=1, random_state=42)
        clarans.fit(self.X)
        labels = clarans.predict(self.X)
        self.assertEqual(labels.shape, (100,))

    def test_convergence(self):
        clarans = CLARANS(n_clusters=3, numlocal=10, maxneighbor=100, random_state=42)
        clarans.fit(self.X)
        score = silhouette_score(self.X, clarans.labels_)
        self.assertGreater(score, -0.1, f"Silhouette score too low: {score}")

    def test_init_methods(self):
        """Test different initialization methods."""
        for init_method in ["random", "heuristic", "k-medoids++", "build"]:
            clarans = CLARANS(
                n_clusters=3,
                numlocal=1,
                maxneighbor=10,
                init=init_method,
                random_state=42,
            )
            clarans.fit(self.X)
            self.assertEqual(len(clarans.cluster_centers_), 3)
            self.assertEqual(len(clarans.labels_), 100)

    def test_precomputed_init(self):
        """Test initialization with precomputed array."""
        init_centers = self.X[[0, 10, 20]]
        clarans = CLARANS(
            n_clusters=3,
            numlocal=1,
            maxneighbor=10,
            init=init_centers,
            random_state=42
        )
        clarans.fit(self.X)
        self.assertEqual(len(clarans.cluster_centers_), 3)

    def test_metrics(self):
        """Test different metrics."""
        for metric in ["euclidean", "manhattan"]:
            clarans = CLARANS(
                n_clusters=3, numlocal=1, maxneighbor=10, metric=metric, random_state=42
            )
            clarans.fit(self.X)
            self.assertEqual(len(clarans.cluster_centers_), 3)

    def test_max_iter(self):
        """Test max_iter parameter."""
        clarans = CLARANS(
            n_clusters=3, numlocal=1, maxneighbor=50, max_iter=5, random_state=42
        )
        clarans.fit(self.X)
        self.assertEqual(len(clarans.cluster_centers_), 3)

    def test_input_validation_init(self):
        """Test invalid init parameter."""
        clarans = CLARANS(n_clusters=3, init="invalid_method")
        with self.assertRaises(ValueError):
            clarans.fit(self.X)

        clarans = CLARANS(n_clusters=3, init=self.X[:2])  
        with self.assertRaises(ValueError):
            clarans.fit(self.X)


if __name__ == "__main__":
    unittest.main()
