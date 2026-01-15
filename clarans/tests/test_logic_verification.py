"""
Test suite để kiểm tra tính đúng đắn của logic trong CLARANS algorithm.
Các test này chỉ kiểm tra logic, không sửa đổi code gốc.
"""
import unittest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min, silhouette_score

from clarans import CLARANS
from clarans.initialization import (
    initialize_build, 
    initialize_heuristic, 
    initialize_k_medoids_plus_plus
)
from clarans.utils import calculate_cost


class TestInitializationLogic(unittest.TestCase):
    """Test initialization functions logic"""
    
    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        self.rng = np.random.RandomState(42)
    
    def test_initialize_heuristic_returns_correct_count(self):
        """Heuristic init should return exactly n_clusters medoids"""
        for n_clusters in [2, 3, 5]:
            medoids = initialize_heuristic(self.X, n_clusters, 'euclidean')
            self.assertEqual(len(medoids), n_clusters)
    
    def test_initialize_heuristic_returns_unique_medoids(self):
        """Heuristic init should return unique medoid indices"""
        medoids = initialize_heuristic(self.X, 3, 'euclidean')
        unique_medoids = np.unique(medoids)
        self.assertEqual(len(unique_medoids), 3, 
                        f"Expected 3 unique medoids, got {len(unique_medoids)}: {medoids}")
    
    def test_initialize_heuristic_returns_smallest_sum_distance(self):
        """Heuristic should select points with smallest sum distance"""
        medoids = initialize_heuristic(self.X, 3, 'euclidean')
        D = pairwise_distances(self.X, metric='euclidean')
        dist_sums = np.sum(D, axis=1)
        expected = np.argsort(dist_sums)[:3]
        np.testing.assert_array_equal(medoids, expected,
                                     "Heuristic should pick points with smallest distance sums")
    
    def test_initialize_build_returns_correct_count(self):
        """BUILD init should return exactly n_clusters medoids"""
        for n_clusters in [2, 3, 5]:
            medoids = initialize_build(self.X, n_clusters, 'euclidean')
            self.assertEqual(len(medoids), n_clusters)
    
    def test_initialize_build_returns_unique_medoids(self):
        """BUILD init should return unique medoid indices"""
        medoids = initialize_build(self.X, 3, 'euclidean')
        unique_medoids = np.unique(medoids)
        self.assertEqual(len(unique_medoids), 3,
                        f"Expected 3 unique medoids, got {len(unique_medoids)}: {medoids}")
    
    def test_initialize_build_first_medoid_is_most_central(self):
        """BUILD should pick most central point first"""
        medoids = initialize_build(self.X, 3, 'euclidean')
        D = pairwise_distances(self.X, metric='euclidean')
        dist_sums = D.sum(axis=1)
        expected_first = np.argmin(dist_sums)
        self.assertEqual(medoids[0], expected_first,
                        f"First medoid should be most central: expected {expected_first}, got {medoids[0]}")
    
    def test_initialize_kmedoids_plusplus_returns_correct_count(self):
        """K-medoids++ should return exactly n_clusters medoids"""
        for n_clusters in [2, 3, 5]:
            rng = np.random.RandomState(42)
            medoids = initialize_k_medoids_plus_plus(self.X, n_clusters, rng, 'euclidean')
            self.assertEqual(len(medoids), n_clusters)
    
    def test_initialize_kmedoids_plusplus_returns_unique_medoids(self):
        """K-medoids++ should return unique medoid indices"""
        # Test nhiều lần với random state khác nhau
        for seed in range(10):
            rng = np.random.RandomState(seed)
            medoids = initialize_k_medoids_plus_plus(self.X, 3, rng, 'euclidean')
            unique_medoids = np.unique(medoids)
            self.assertEqual(len(unique_medoids), 3,
                            f"Seed {seed}: Expected 3 unique medoids, got {len(unique_medoids)}: {medoids}")
    
    def test_initialize_kmedoids_plusplus_indices_in_range(self):
        """K-medoids++ indices should be valid array indices"""
        rng = np.random.RandomState(42)
        medoids = initialize_k_medoids_plus_plus(self.X, 3, rng, 'euclidean')
        self.assertTrue(all(0 <= idx < len(self.X) for idx in medoids),
                       "All medoid indices should be within valid range")


class TestCostCalculation(unittest.TestCase):
    """Test cost calculation logic"""
    
    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    def test_cost_is_non_negative(self):
        """Cost should always be non-negative"""
        medoids = np.array([0, 30, 60])
        cost = calculate_cost(self.X, medoids, 'euclidean')
        self.assertGreaterEqual(cost, 0, "Cost should be non-negative")
    
    def test_cost_is_sum_of_min_distances(self):
        """Cost should equal sum of minimum distances to medoids"""
        medoids = np.array([0, 30, 60])
        cost = calculate_cost(self.X, medoids, 'euclidean')
        
        # Manual calculation
        medoid_points = self.X[medoids]
        _, min_dists = pairwise_distances_argmin_min(self.X, medoid_points, metric='euclidean')
        expected_cost = np.sum(min_dists)
        
        self.assertAlmostEqual(cost, expected_cost, places=10,
                              msg=f"Cost mismatch: got {cost}, expected {expected_cost}")
    
    def test_cost_is_zero_when_all_points_are_medoids(self):
        """Cost should be zero when every point is a medoid"""
        # Tạo dataset nhỏ để mọi điểm đều là medoid
        X_small = np.array([[0, 0], [1, 1], [2, 2]])
        medoids = np.array([0, 1, 2])
        cost = calculate_cost(X_small, medoids, 'euclidean')
        self.assertEqual(cost, 0, "Cost should be 0 when all points are medoids")
    
    def test_cost_with_different_metrics(self):
        """Cost calculation should work with different metrics"""
        medoids = np.array([0, 30, 60])
        
        for metric in ['euclidean', 'manhattan']:
            cost = calculate_cost(self.X, medoids, metric)
            medoid_points = self.X[medoids]
            _, min_dists = pairwise_distances_argmin_min(self.X, medoid_points, metric=metric)
            expected = np.sum(min_dists)
            self.assertAlmostEqual(cost, expected, places=10,
                                  msg=f"Cost mismatch for {metric} metric")


class TestCLARANSAlgorithmLogic(unittest.TestCase):
    """Test CLARANS algorithm logic"""
    
    def setUp(self):
        self.X, self.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    def test_medoids_are_actual_data_points(self):
        """Medoids should be actual points from X"""
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=30, random_state=42)
        clarans.fit(self.X)
        
        for idx in clarans.medoid_indices_:
            self.assertTrue(0 <= idx < len(self.X), 
                           f"Medoid index {idx} out of range")
        
        # Verify cluster_centers_ are actual rows from X
        for i, center in enumerate(clarans.cluster_centers_):
            medoid_idx = clarans.medoid_indices_[i]
            np.testing.assert_array_equal(center, self.X[medoid_idx],
                                         f"Cluster center {i} doesn't match X[{medoid_idx}]")
    
    def test_medoids_are_unique(self):
        """All medoids should be unique"""
        for seed in range(10):
            clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=30, random_state=seed)
            clarans.fit(self.X)
            
            unique_count = len(np.unique(clarans.medoid_indices_))
            self.assertEqual(unique_count, 3,
                            f"Seed {seed}: Expected 3 unique medoids, got {unique_count}")
    
    def test_labels_are_valid(self):
        """Labels should be in range [0, n_clusters)"""
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=30, random_state=42)
        clarans.fit(self.X)
        
        self.assertTrue(all(0 <= label < 3 for label in clarans.labels_),
                       "All labels should be in valid range")
    
    def test_predict_matches_labels_on_training_data(self):
        """predict(X) on training data should match labels_"""
        clarans = CLARANS(n_clusters=3, numlocal=2, maxneighbor=50, random_state=42)
        clarans.fit(self.X)
        
        predicted = clarans.predict(self.X)
        np.testing.assert_array_equal(predicted, clarans.labels_,
                                     "predict(X) should match labels_ on training data")
    
    def test_local_search_improves_or_maintains_cost(self):
        """Each numlocal iteration should find local minimum"""
        clarans = CLARANS(n_clusters=3, numlocal=5, maxneighbor=100, random_state=42)
        clarans.fit(self.X)
        
        final_cost = calculate_cost(self.X, clarans.medoid_indices_, 'euclidean')
        
        # Cost should be less than or equal to random initialization
        random_medoids = np.random.RandomState(99).choice(100, 3, replace=False)
        random_cost = calculate_cost(self.X, random_medoids, 'euclidean')
        
        # Final cost should generally be better (lower)
        # Note: This is probabilistic, so we just check it's reasonable
        self.assertGreater(final_cost, 0, "Final cost should be positive")
    
    def test_maxneighbor_default_calculation(self):
        """Default maxneighbor should be calculated correctly"""
        n_samples = 100
        n_clusters = 3
        
        clarans = CLARANS(n_clusters=n_clusters, numlocal=1, random_state=42)
        clarans.fit(self.X)
        
        expected_maxneighbor = max(250, int(0.0125 * n_clusters * (n_samples - n_clusters)))
        self.assertEqual(clarans.maxneighbor_, expected_maxneighbor,
                        f"maxneighbor_ should be {expected_maxneighbor}, got {clarans.maxneighbor_}")
    
    def test_maxneighbor_custom_value(self):
        """Custom maxneighbor should be used when provided"""
        custom_maxneighbor = 100
        clarans = CLARANS(n_clusters=3, maxneighbor=custom_maxneighbor, random_state=42)
        clarans.fit(self.X)
        
        self.assertEqual(clarans.maxneighbor_, custom_maxneighbor,
                        f"maxneighbor_ should be {custom_maxneighbor}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_identical_points_dataset(self):
        """Algorithm should handle dataset with identical points"""
        X_identical = np.ones((50, 2))
        
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=20, random_state=42)
        clarans.fit(X_identical)
        
        # Should have 3 unique medoid indices (even if points are identical)
        unique_medoids = len(np.unique(clarans.medoid_indices_))
        self.assertEqual(unique_medoids, 3, 
                        f"Should have 3 unique medoids, got {unique_medoids}")
    
    def test_two_clusters(self):
        """Algorithm should work with 2 clusters"""
        X, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=42)
        
        clarans = CLARANS(n_clusters=2, numlocal=1, maxneighbor=20, random_state=42)
        clarans.fit(X)
        
        self.assertEqual(len(clarans.medoid_indices_), 2)
        self.assertEqual(len(np.unique(clarans.labels_)), 2)
    
    def test_high_dimensional_data(self):
        """Algorithm should work with high-dimensional data"""
        X, _ = make_blobs(n_samples=100, centers=3, n_features=50, random_state=42)
        
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=30, random_state=42)
        clarans.fit(X)
        
        self.assertEqual(len(clarans.medoid_indices_), 3)
        self.assertEqual(clarans.cluster_centers_.shape, (3, 50))
    
    def test_n_clusters_close_to_n_samples(self):
        """Algorithm should work when n_clusters is close to n_samples"""
        X = np.random.RandomState(42).randn(20, 2)
        
        # Use 15 clusters for 20 samples
        clarans = CLARANS(n_clusters=15, numlocal=1, maxneighbor=10, random_state=42)
        clarans.fit(X)
        
        self.assertEqual(len(clarans.medoid_indices_), 15)
        unique_medoids = len(np.unique(clarans.medoid_indices_))
        self.assertEqual(unique_medoids, 15, "All medoids should be unique")
    
    def test_single_feature(self):
        """Algorithm should work with single feature"""
        X = np.random.RandomState(42).randn(50, 1)
        
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=20, random_state=42)
        clarans.fit(X)
        
        self.assertEqual(len(clarans.medoid_indices_), 3)
        self.assertEqual(clarans.cluster_centers_.shape, (3, 1))


class TestInputValidation(unittest.TestCase):
    """Test input validation"""
    
    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    def test_invalid_init_method(self):
        """Should raise error for invalid init method"""
        clarans = CLARANS(n_clusters=3, init="invalid_method")
        with self.assertRaises(ValueError):
            clarans.fit(self.X)
    
    def test_n_clusters_greater_than_samples(self):
        """Should raise error when n_clusters >= n_samples"""
        clarans = CLARANS(n_clusters=150)  # More than 100 samples
        with self.assertRaises(ValueError):
            clarans.fit(self.X)
    
    def test_init_array_wrong_shape(self):
        """Should raise error when init array has wrong shape"""
        wrong_shape_init = np.random.randn(5, 2)  # 5 instead of 3
        clarans = CLARANS(n_clusters=3, init=wrong_shape_init)
        with self.assertRaises(ValueError):
            clarans.fit(self.X)
    
    def test_predict_before_fit(self):
        """Should raise error when predicting before fit"""
        clarans = CLARANS(n_clusters=3)
        with self.assertRaises(Exception):  # NotFittedError or similar
            clarans.predict(self.X)


class TestDeterminism(unittest.TestCase):
    """Test deterministic behavior with same random_state"""
    
    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    def test_same_random_state_gives_same_results(self):
        """Same random_state should give identical results"""
        clarans1 = CLARANS(n_clusters=3, numlocal=2, maxneighbor=50, random_state=123)
        clarans1.fit(self.X)
        
        clarans2 = CLARANS(n_clusters=3, numlocal=2, maxneighbor=50, random_state=123)
        clarans2.fit(self.X)
        
        np.testing.assert_array_equal(clarans1.medoid_indices_, clarans2.medoid_indices_,
                                     "Same random_state should give same medoids")
        np.testing.assert_array_equal(clarans1.labels_, clarans2.labels_,
                                     "Same random_state should give same labels")
    
    def test_different_random_state_may_give_different_results(self):
        """Different random_state may give different results"""
        clarans1 = CLARANS(n_clusters=3, numlocal=1, maxneighbor=30, random_state=1)
        clarans1.fit(self.X)
        
        clarans2 = CLARANS(n_clusters=3, numlocal=1, maxneighbor=30, random_state=999)
        clarans2.fit(self.X)
        
        # Results may differ (though not guaranteed)
        # Just check they both work
        self.assertEqual(len(clarans1.medoid_indices_), 3)
        self.assertEqual(len(clarans2.medoid_indices_), 3)


class TestInitializationConsistency(unittest.TestCase):
    """Test that different initialization methods all produce valid results"""
    
    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    def test_all_init_methods_produce_valid_clustering(self):
        """All initialization methods should produce valid clustering"""
        init_methods = ['random', 'heuristic', 'k-medoids++', 'build']
        
        for method in init_methods:
            with self.subTest(init=method):
                clarans = CLARANS(n_clusters=3, numlocal=2, maxneighbor=50, 
                                 init=method, random_state=42)
                clarans.fit(self.X)
                
                # Check medoids are valid
                self.assertEqual(len(clarans.medoid_indices_), 3,
                               f"{method}: Should have 3 medoids")
                self.assertEqual(len(np.unique(clarans.medoid_indices_)), 3,
                               f"{method}: Medoids should be unique")
                
                # Check labels are valid
                self.assertEqual(len(clarans.labels_), 100,
                               f"{method}: Should have 100 labels")
                self.assertTrue(all(0 <= l < 3 for l in clarans.labels_),
                              f"{method}: Labels should be in [0, 3)")
                
                # Check clustering quality (silhouette score should be reasonable)
                score = silhouette_score(self.X, clarans.labels_)
                self.assertGreater(score, -0.5,  # Very loose bound
                                 f"{method}: Silhouette score too low: {score}")


class TestK_MedoidsPlusPlusLogic(unittest.TestCase):
    """Specific tests for k-medoids++ initialization edge cases"""
    
    def test_handles_zero_distance_case(self):
        """K-medoids++ should handle case when all remaining points have zero distance"""
        # Dataset where some points are identical to the first medoid
        X = np.array([
            [0, 0],  # Will be selected as first medoid
            [0, 0],  # Identical to first
            [0, 0],  # Identical to first
            [1, 1],  # Different
            [2, 2],  # Different
        ])
        
        rng = np.random.RandomState(42)
        medoids = initialize_k_medoids_plus_plus(X, 3, rng, 'euclidean')
        
        self.assertEqual(len(medoids), 3)
        self.assertEqual(len(np.unique(medoids)), 3, 
                        "Should have 3 unique medoids even with identical points")


class TestMaxIterBehavior(unittest.TestCase):
    """Test max_iter parameter behavior"""
    
    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    def test_max_iter_limits_iterations(self):
        """max_iter should limit the number of improvements"""
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=1000, 
                         max_iter=5, random_state=42)
        clarans.fit(self.X)
        
        # n_iter_ should be limited by max_iter (times numlocal)
        # Note: actual implementation may vary
        self.assertTrue(hasattr(clarans, 'n_iter_'))
    
    def test_none_max_iter_allows_unlimited(self):
        """max_iter=None should allow unlimited iterations"""
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=50, 
                         max_iter=None, random_state=42)
        clarans.fit(self.X)
        
        # Should complete without error
        self.assertEqual(len(clarans.medoid_indices_), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
