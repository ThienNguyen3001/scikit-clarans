"""
Comprehensive test suite to verify CLARANS algorithm logic before making fixes.
Tests cover edge cases, performance issues, and correctness of all components.
"""
import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min
from clarans import CLARANS
from clarans.initialization import (initialize_build, initialize_heuristic, 
                                    initialize_k_medoids_plus_plus)
from clarans.utils import calculate_cost


def test_1_basic_functionality():
    """Test 1: Basic CLARANS functionality"""
    print("\n" + "="*70)
    print("TEST 1: Basic Functionality")
    print("="*70)
    
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    clarans = CLARANS(n_clusters=3, numlocal=2, maxneighbor=50, random_state=42)
    clarans.fit(X)
    
    # Check basic attributes
    assert hasattr(clarans, 'medoid_indices_'), "Missing medoid_indices_"
    assert hasattr(clarans, 'cluster_centers_'), "Missing cluster_centers_"
    assert hasattr(clarans, 'labels_'), "Missing labels_"
    
    assert len(clarans.medoid_indices_) == 3, f"Expected 3 medoids, got {len(clarans.medoid_indices_)}"
    assert len(clarans.labels_) == 100, f"Expected 100 labels, got {len(clarans.labels_)}"
    
    print(f"[OK] Medoid indices: {clarans.medoid_indices_}")
    print(f"[OK] Number of iterations: {clarans.n_iter_}")
    print(f"[OK] Unique labels: {np.unique(clarans.labels_)}")
    print("[OK] TEST 1 PASSED")


def test_2_medoid_uniqueness():
    """Test 2: Verify all medoids are unique"""
    print("\n" + "="*70)
    print("TEST 2: Medoid Uniqueness")
    print("="*70)
    
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    failed_runs = []
    for i in range(10):
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=30, random_state=i)
        clarans.fit(X)
        
        unique_medoids = len(np.unique(clarans.medoid_indices_))
        if unique_medoids != 3:
            failed_runs.append({
                'run': i,
                'medoids': clarans.medoid_indices_,
                'unique': unique_medoids
            })
            print(f"Run {i}: Found duplicate medoids! {clarans.medoid_indices_}")
        else:
            print(f"Run {i}: All medoids unique {clarans.medoid_indices_}")
    
    if failed_runs:
        print(f"\nTEST 2 FAILED: {len(failed_runs)}/10 runs had duplicate medoids")
        for fail in failed_runs:
            print(f"  Run {fail['run']}: {fail['medoids']} (only {fail['unique']} unique)")
    assert not failed_runs, "Some runs produced duplicate medoids"
    print("\nTEST 2 PASSED: All 10 runs had unique medoids")


def test_3_initialization_methods():
    """Test 3: Test all initialization methods"""
    print("\n" + "="*70)
    print("TEST 3: Initialization Methods")
    print("="*70)
    
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    init_methods = ['random', 'heuristic', 'k-medoids++', 'build']
    
    results = {}
    for method in init_methods:
        print(f"\nTesting init='{method}'...")
        try:
            clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=30, 
                            init=method, random_state=42)
            clarans.fit(X)
            
            medoids = clarans.medoid_indices_
            unique_count = len(np.unique(medoids))
            
            results[method] = {
                'success': True,
                'medoids': medoids,
                'unique': unique_count,
                'error': None
            }
            
            if unique_count == 3:
                print(f"  Success! Medoids: {medoids}")
            else:
                print(f"  Warning! Only {unique_count} unique medoids: {medoids}")
                
        except Exception as e:
            results[method] = {
                'success': False,
                'medoids': None,
                'unique': 0,
                'error': str(e)
            }
            print(f"  ERROR: {e}")
    
    # Summary
    print(f"\n{'Method':<20} {'Success':<10} {'Unique Medoids':<15} {'Status'}")
    print("-" * 70)
    for method, result in results.items():
        if result['success']:
            status = "PASS" if result['unique'] == 3 else "WARN"
            print(f"{method:<20} {'Yes':<10} {result['unique']:<15} {status}")
        else:
            print(f"{method:<20} {'No':<10} {'N/A':<15} FAIL")
    
    all_passed = all(r['success'] and r['unique'] == 3 for r in results.values())
    if not all_passed:
        print("\n[FAIL] TEST 3 FAILED")
    assert all_passed, "Some initialization methods failed or produced duplicate medoids"
    print("\n[OK] TEST 3 PASSED")


def test_4_initialization_uniqueness():
    """Test 4: Check if initialization functions return unique medoids"""
    print("\n" + "="*70)
    print("TEST 4: Initialization Functions Return Unique Medoids")
    print("="*70)
    
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    rng = np.random.RandomState(42)
    
    results = {}
    
    # Test BUILD
    print("\nTesting initialize_build()...")
    for i in range(5):
        medoids = initialize_build(X, 3, 'euclidean')
        unique = len(np.unique(medoids))
        print(f"  Run {i}: {medoids} -> {unique} unique")
        results.setdefault('build', []).append(unique == 3)
    
    # Test Heuristic
    print("\nTesting initialize_heuristic()...")
    for i in range(5):
        medoids = initialize_heuristic(X, 3, 'euclidean')
        unique = len(np.unique(medoids))
        print(f"  Run {i}: {medoids} -> {unique} unique")
        results.setdefault('heuristic', []).append(unique == 3)
    
    # Test K-medoids++
    print("\nTesting initialize_k_medoids_plus_plus()...")
    for i in range(5):
        rng = np.random.RandomState(i)
        medoids = initialize_k_medoids_plus_plus(X, 3, rng, 'euclidean')
        unique = len(np.unique(medoids))
        print(f"  Run {i}: {medoids} -> {unique} unique")
        results.setdefault('k-medoids++', []).append(unique == 3)
    
    # Summary
    print(f"\n{'Method':<20} {'Success Rate':<15} {'Status'}")
    print("-" * 50)
    all_passed = True
    for method, passes in results.items():
        success_rate = f"{sum(passes)}/{len(passes)}"
        status = "PASS" if all(passes) else "FAIL"
        print(f"{method:<20} {success_rate:<15} {status}")
        if not all(passes):
            all_passed = False
    
    if not all_passed:
        print("\n[FAIL] TEST 4 FAILED: Some initialization methods produce duplicate medoids")
    assert all_passed, "Some initialization functions produced duplicate medoids"
    print("\n[OK] TEST 4 PASSED")


def test_5_cost_calculation():
    """Test 5: Verify cost calculation is correct"""
    print("\n" + "="*70)
    print("TEST 5: Cost Calculation Correctness")
    print("="*70)
    
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    # Test with known medoids
    test_cases = [
        np.array([0, 30, 60]),
        np.array([10, 40, 70]),
        np.array([5, 50, 95])
    ]
    
    all_correct = True
    for i, medoids in enumerate(test_cases):
        cost = calculate_cost(X, medoids, 'euclidean')
        
        # Manual calculation
        medoid_points = X[medoids]
        _, min_dists = pairwise_distances_argmin_min(X, medoid_points, metric='euclidean')
        expected_cost = np.sum(min_dists)
        
        match = np.isclose(cost, expected_cost)
        status = "OK" if match else "FAIL"
        print(f"{status} Test case {i+1}: calculated={cost:.4f}, expected={expected_cost:.4f}, match={match}")
        
        if not match:
            all_correct = False
    
    if not all_correct:
        print("\n[FAIL] TEST 5 FAILED")
    assert all_correct, "Cost calculation mismatch for one or more test cases"
    print("\n[OK] TEST 5 PASSED")


def test_6_edge_case_all_same_points():
    """Test 6: Edge case - all points are identical"""
    print("\n" + "="*70)
    print("TEST 6: Edge Case - Identical Points")
    print("="*70)
    
    # Create dataset with all identical points
    X = np.ones((50, 2))
    
    try:
        clarans = CLARANS(n_clusters=3, numlocal=1, maxneighbor=20, random_state=42)
        clarans.fit(X)
        
        unique_medoids = len(np.unique(clarans.medoid_indices_))
        print(f"Medoids: {clarans.medoid_indices_}")
        print(f"Unique medoids: {unique_medoids}")
        
        if unique_medoids != 3:
            print(f"[FAIL] TEST 6 FAILED: Only {unique_medoids} unique medoids for identical points")
        assert unique_medoids == 3, "Expected unique medoids even when all points are identical"
        print("[OK] TEST 6 PASSED: Handled identical points correctly")
            
    except Exception as e:
        print(f"[FAIL] TEST 6 FAILED with exception: {e}")
        raise


def test_7_candidate_selection_performance():
    """Test 7: Check performance of candidate selection in main loop"""
    print("\n" + "="*70)
    print("TEST 7: Candidate Selection Performance")
    print("="*70)
    
    # Test with increasing cluster sizes
    test_configs = [
        (100, 3),
        (100, 10),
        (100, 20),
        (200, 30),
    ]
    
    for n_samples, n_clusters in test_configs:
        X, y = make_blobs(n_samples=n_samples, centers=n_clusters, 
                         n_features=2, random_state=42)
        
        print(f"\nTesting n_samples={n_samples}, n_clusters={n_clusters}...")
        
        start_time = time.time()
        clarans = CLARANS(n_clusters=n_clusters, numlocal=1, maxneighbor=50, 
                         random_state=42)
        clarans.fit(X)
        elapsed = time.time() - start_time
        
        print(f"  Time: {elapsed:.3f}s, Iterations: {clarans.n_iter_}")
        print(f"  Medoids: {clarans.medoid_indices_[:5]}{'...' if n_clusters > 5 else ''}")
        
        if elapsed > 10:  # Warning if too slow
            print(f"  WARNING: Slow performance ({elapsed:.1f}s)")
    
    print("\n[OK] TEST 7 COMPLETED")


def test_8_max_iter_parameter():
    """Test 8: Verify max_iter parameter works correctly"""
    print("\n" + "="*70)
    print("TEST 8: max_iter Parameter")
    print("="*70)
    
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    # Test different max_iter values
    max_iter_values = [1, 5, 10, 50, None]
    
    for max_iter in max_iter_values:
        clarans = CLARANS(n_clusters=3, numlocal=2, maxneighbor=100, 
                         max_iter=max_iter, random_state=42)
        clarans.fit(X)
        
        print(f"max_iter={str(max_iter):<6} -> n_iter_={clarans.n_iter_}")
        
        if max_iter is not None and clarans.n_iter_ > max_iter * 2:  # 2 locals
            print(f"  WARNING: n_iter_ ({clarans.n_iter_}) exceeds expected max ({max_iter * 2})")
    
    print("\n[OK] TEST 8 COMPLETED")


def test_9_predict_consistency():
    """Test 9: Verify predict() returns consistent results"""
    print("\n" + "="*70)
    print("TEST 9: Predict Consistency")
    print("="*70)
    
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    clarans = CLARANS(n_clusters=3, numlocal=2, maxneighbor=50, random_state=42)
    clarans.fit(X)
    
    # Predict on training data
    labels_predict = clarans.predict(X)
    labels_fit = clarans.labels_
    
    match = np.array_equal(labels_predict, labels_fit)
    
    if match:
        print("[OK] Predict on training data matches fit labels")
    else:
        diff_count = np.sum(labels_predict != labels_fit)
        print(f"[FAIL] Predict differs from fit in {diff_count} points")
        print(f"  First 10 predict: {labels_predict[:10]}")
        print(f"  First 10 fit:     {labels_fit[:10]}")
    
    # Predict on new data
    X_new, _ = make_blobs(n_samples=20, centers=3, n_features=2, random_state=99)
    labels_new = clarans.predict(X_new)
    
    print(f"[OK] Predict on new data: {labels_new[:10]}")
    print(f"  All labels in range [0, {clarans.n_clusters-1}]: {np.all((labels_new >= 0) & (labels_new < clarans.n_clusters))}")
    
    if not match:
        print("\n[FAIL] TEST 9 FAILED")
    assert match, "predict(X) on training data should match labels_ from fit"
    print("\n[OK] TEST 9 PASSED")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*70)
    print("CLARANS ALGORITHM LOGIC TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Functionality", test_1_basic_functionality),
        ("Medoid Uniqueness", test_2_medoid_uniqueness),
        ("Initialization Methods", test_3_initialization_methods),
        ("Init Function Uniqueness", test_4_initialization_uniqueness),
        ("Cost Calculation", test_5_cost_calculation),
        ("Edge Case - Identical Points", test_6_edge_case_all_same_points),
        ("Candidate Selection Performance", test_7_candidate_selection_performance),
        ("Max Iter Parameter", test_8_max_iter_parameter),
        ("Predict Consistency", test_9_predict_consistency),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result if result is not None else True
        except Exception as e:
            print(f"\n[CRASH] {test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print("-" * 70)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("\nALL TESTS PASSED! Algorithm logic is correct.")
    else:
        print(f"\nWARNING: {failed} TEST(S) FAILED! Issues found that need fixing.")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
