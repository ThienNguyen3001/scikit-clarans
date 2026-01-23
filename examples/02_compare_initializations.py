"""
02_compare_initializations.py
=============================
Compare different initialization strategies available in CLARANS and report
final clustering cost for each method.

Run with: python examples/02_compare_initializations.py
"""

import time

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from clarans import CLARANS
from clarans.utils import calculate_cost


def main():
    X, _ = make_blobs(n_samples=800, centers=4, n_features=2, random_state=1)
    inits = ["random", "k-medoids++", "heuristic", "build"]

    results = []
    for init in inits:
        t0 = time.time()
        model = CLARANS(n_clusters=4, numlocal=3, init=init, random_state=42)
        model.fit(X)
        t1 = time.time()

        cost = calculate_cost(X, model.medoid_indices_, metric=model.metric)
        results.append((init, cost, t1 - t0))

        print(f"init={init:12s}  cost={cost:.2f}  time={t1-t0:.3f}s")

    # Simple bar plots
    inits, costs, times = zip(*results)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(inits, costs, color="C1")
    plt.title("Final clustering cost by init method")
    plt.ylabel("Cost")

    plt.subplot(1, 2, 2)
    plt.bar(inits, times, color="C2")
    plt.title("Runtime by init method")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
