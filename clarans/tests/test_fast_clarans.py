import unittest
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from clarans import FastCLARANS


class TestFastCLARANS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    def test_n_iter_and_n_swaps_attributes(self):
        """Test that n_iter_, n_swaps_, and neighborhood_size_ are set correctly in FastCLARANS,
        and deprecated total_n_iter_ / total_n_swaps_ / total_neighbors_ / n_neighbors_ are removed.
        """
        # Multi-restart run
        model = FastCLARANS(n_clusters=3, num_local=2, max_neighbors=50, random_state=42)
        model.fit(self.X)
        self.assertTrue(hasattr(model, "n_iter_"))
        self.assertTrue(hasattr(model, "n_swaps_"))
        self.assertTrue(hasattr(model, "neighborhood_size_"))
        self.assertFalse(hasattr(model, "n_neighbors_"))
        self.assertFalse(hasattr(model, "total_neighbors_"))
        self.assertFalse(hasattr(model, "total_n_iter_"))
        self.assertFalse(hasattr(model, "total_n_swaps_"))

        self.assertGreaterEqual(model.n_iter_, 1)
        self.assertGreaterEqual(model.n_swaps_, 0)
        self.assertLessEqual(model.n_swaps_, model.n_iter_)

        # neighborhood_size_ check: n - k candidate points
        n_samples = self.X.shape[0]
        self.assertEqual(model.neighborhood_size_, n_samples - 3)

        # Single-restart run
        m_single = FastCLARANS(n_clusters=3, num_local=1, max_neighbors=50, random_state=42)
        m_single.fit(self.X)
        self.assertGreaterEqual(m_single.n_iter_, 1)
        self.assertLessEqual(m_single.n_swaps_, m_single.n_iter_)
        self.assertLessEqual(model.inertia_, m_single.inertia_)

    def test_single_cluster_1d_swap(self):
        """Test on 1D data where initial point is suboptimal to verify swap occurs."""
        X_1d = np.array([[0.0], [1.0], [10.0]])
        # random_state=0 initially selects index 2 (10.0) with cost 19.0
        model_1d = FastCLARANS(n_clusters=1, num_local=1, max_neighbors=10, random_state=0)
        model_1d.fit(X_1d)
        self.assertEqual(model_1d.medoid_indices_[0], 1)
        self.assertAlmostEqual(model_1d.inertia_, 10.0)

    def test_max_neighbors_default(self):
        """Default max_neighbors should be None and calculated correctly in FastCLARANS."""
        model = FastCLARANS(n_clusters=3, num_local=1, random_state=42)
        self.assertIsNone(model.max_neighbors)
        model.fit(self.X)
        expected = max(1, int(250 / 3), int(0.025 * (100 - 3)))
        self.assertEqual(model.max_neighbors_, expected)

    def test_max_neighbors_explicit_none(self):
        """Explicit max_neighbors=None should work without any deprecation warning."""
        model = FastCLARANS(n_clusters=3, num_local=1, max_neighbors=None, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model.fit(self.X)
        expected = max(1, int(250 / 3), int(0.025 * (100 - 3)))
        self.assertEqual(model.max_neighbors_, expected)

    def test_max_neighbors_explicit_auto(self):
        """Explicit max_neighbors='auto' should raise a FutureWarning and calculate correctly."""
        model = FastCLARANS(n_clusters=3, num_local=1, max_neighbors="auto", random_state=42)
        with pytest.warns(FutureWarning, match="deprecated"):
            model.fit(self.X)
        expected = max(1, int(250 / 3), int(0.025 * (100 - 3)))
        self.assertEqual(model.max_neighbors_, expected)

    def test_no_cache_parameter(self):
        """FastCLARANS should not accept cache or cost_evaluation parameter in __init__."""
        with self.assertRaises(TypeError):
            FastCLARANS(cache=False)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            FastCLARANS(cost_evaluation="brute_force")  # type: ignore[call-arg]
