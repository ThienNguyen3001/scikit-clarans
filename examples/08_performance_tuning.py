"""
08_performance_tuning.py
========================
Show the trade-off between parameter choices (max_neighbors, num_local) and
runtime / final cost.

Run: python examples/08_performance_tuning.py
"""

import time

from sklearn.datasets import make_blobs

from clarans import CLARANS
from clarans.utils import calculate_cost


def main():
    X, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=42)

    combinations = [
        {"num_local": 1, "max_neighbors": 250},
        {"num_local": 3, "max_neighbors": 250},
        {"num_local": 3, "max_neighbors": 1000},
    ]

    for c in combinations:
        t0 = time.time()
        model = CLARANS(
            n_clusters=5,
            num_local=c["num_local"],
            max_neighbors=c["max_neighbors"],
            random_state=0,
        )
        model.fit(X)
        t1 = time.time()

        cost = calculate_cost(X, model.medoid_indices_)
        msg = (
            f"num_local={c['num_local']:2d}  "
            f"max_neighbors={c['max_neighbors']:4d}  "
            f"time={t1-t0:.3f}s  "
            f"cost={cost:.2f}"
        )
        print(msg)


if __name__ == "__main__":
    main()
