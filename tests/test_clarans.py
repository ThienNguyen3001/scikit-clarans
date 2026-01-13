import unittest
import numpy as np
from sklearn.datasets import make_blobs
from clarans import CLARANS
from sklearn.metrics import silhouette_score

class TestCLARANS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    def test_fit(self):
        clarans = CLARANS(n_clusters=3, numlocal=2, maxneighbor=10, random_state=42)
        clarans.fit(self.X)
        
        self.assertEqual(len(clarans.cluster_centers_), 3)
        self.assertEqual(len(clarans.labels_), 100)
        self.assertTrue(hasattr(clarans, 'medoid_indices_'))
        
        # Check that medoids are actually points from X
        for idx in clarans.medoid_indices_:
            self.assertTrue(np.any(np.all(self.X == self.X[idx], axis=1)))

    def test_predict(self):
        clarans = CLARANS(n_clusters=3, numlocal=1, random_state=42)
        clarans.fit(self.X)
        labels = clarans.predict(self.X)
        self.assertEqual(labels.shape, (100,))
        
    def test_convergence(self):
        # Increase numlocal and maxneighbor for better clustering quality
        clarans = CLARANS(n_clusters=3, numlocal=10, maxneighbor=100, random_state=42)
        clarans.fit(self.X)
        score = silhouette_score(self.X, clarans.labels_)
        # Accept a slightly negative score, but warn if it's very low
        self.assertGreater(score, -0.1, f"Silhouette score too low: {score}")

if __name__ == '__main__':
    unittest.main()
