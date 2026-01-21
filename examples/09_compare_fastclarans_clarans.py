import time
import tracemalloc
import warnings

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from clarans import CLARANS, FastCLARANS


def run_algorithm_profile(name, model, X):
    """Run model.fit(X) and measure runtime and peak memory usage."""
    print(f"\n--- Running {name} ---")

    tracemalloc.start()
    start_time = time.time()

    model.fit(X)

    end_time = time.time()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_taken = end_time - start_time
    peak_mb = peak_memory / (1024 * 1024)

    print(f"   Time: {time_taken:.4f} s")
    print(f"   Peak memory: {peak_mb:.2f} MB")

    return time_taken, peak_mb, model.medoid_indices_


def calculate_total_cost(X, medoid_indices):
    """Compute total clustering cost (sum of distances to assigned medoid)."""
    medoids = X[medoid_indices]
    dists = pairwise_distances(X, medoids)
    min_dists = np.min(dists, axis=1)
    return np.sum(min_dists)


def run_benchmark():
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("BENCHMARK: SPEED vs MEMORY (CLARANS vs FastCLARANS)")
    print("=" * 60)

    N_SAMPLES = 10000
    N_CLUSTERS = 20
    N_FEATURES = 20
    RANDOM_STATE = 42

    print(f"Data: {N_SAMPLES} samples, {N_FEATURES} features.")
    print(f"Configuration: k={N_CLUSTERS}, random_state={RANDOM_STATE}")
    print(
        f"Estimated full distance matrix size: ~{N_SAMPLES**2 * 8 / (1024**2):.1f} MB"
    )

    X, _ = make_blobs(
        n_samples=N_SAMPLES,
        centers=N_CLUSTERS,
        n_features=N_FEATURES,
        random_state=RANDOM_STATE,
    )

    clarans = CLARANS(n_clusters=N_CLUSTERS, numlocal=1, random_state=RANDOM_STATE)
    time_orig, mem_orig, medoids_orig = run_algorithm_profile(
        "CLARANS (Original)", clarans, X
    )
    cost_orig = calculate_total_cost(X, medoids_orig)

    fast_clarans = FastCLARANS(
        n_clusters=N_CLUSTERS, numlocal=1, random_state=RANDOM_STATE
    )
    time_fast, mem_fast, medoids_fast = run_algorithm_profile(
        "FastCLARANS", fast_clarans, X
    )
    cost_fast = calculate_total_cost(X, medoids_fast)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    speedup = time_orig / time_fast if time_fast > 0 else float("inf")
    print("Runtime:")
    print(f"  - CLARANS:     {time_orig:.4f}s")
    print(f"  - FastCLARANS: {time_fast:.4f}s")
    print(f"  -> FastCLARANS is {speedup:.1f}x faster")

    mem_diff = (mem_fast / mem_orig) if mem_orig > 0 else float("inf")
    print("\nMemory (peak):")
    print(f"  - CLARANS:     {mem_orig:.2f} MB")
    print(f"  - FastCLARANS: {mem_fast:.2f} MB")
    print(
        f"  -> FastCLARANS uses {mem_diff:.1f}x more memory (due to full distance caching)"
    )

    print("\nQuality (Cost - lower is better):")
    print(f"  - CLARANS:     {cost_orig:.2f}")
    print(f"  - FastCLARANS: {cost_fast:.2f}")

    if cost_fast <= cost_orig:
        print("  -> Conclusion: FastCLARANS is equal or better in quality.")
    else:
        print(
            "  -> Conclusion: CLARANS found a slightly better solution (due to randomness)."
        )

    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
