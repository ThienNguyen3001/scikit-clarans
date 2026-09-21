"""Benchmark execution time and peak memory consumption for FastCLARANS vs CLARANS."""

import time
import tracemalloc
from sklearn.datasets import make_blobs

from clarans import CLARANS, FastCLARANS


def run_benchmark():
    """Benchmark execution time and memory consumption for FastCLARANS vs CLARANS."""
    print("\n" + "=" * 70)
    print("ACCELERATION BENCHMARK: FastCLARANS vs CLARANS")
    print("=" * 70)

    for n_samples in [2000, 5000]:
        n_features = 8
        n_clusters = 5
        X, _ = make_blobs(
            n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42
        )

        print(f"\n--- Dataset: N = {n_samples:,}, Features = {n_features}, "
              f"Clusters = {n_clusters} ---")

        for ModelClass in [CLARANS, FastCLARANS]:
            tracemalloc.start()
            t0 = time.perf_counter()
            m = ModelClass(n_clusters=n_clusters, num_local=4, max_neighbors=100, random_state=42)
            m.fit(X)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"{ModelClass.__name__:<15} | Time: {elapsed*1000:6.1f} ms | "
                  f"Peak RAM: {peak/(1024**2):.2f} MB | Inertia: {m.inertia_:.2f}")


if __name__ == "__main__":
    run_benchmark()
