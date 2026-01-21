"""
08_performance_tuning.py
========================
Show the trade-off between parameter choices (maxneighbor, numlocal) and
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
        {"numlocal": 1, "maxneighbor": 250},
        {"numlocal": 3, "maxneighbor": 250},
        {"numlocal": 3, "maxneighbor": 1000},
    ]

    for c in combinations:
        t0 = time.time()
        model = CLARANS(
            n_clusters=5,
            numlocal=c["numlocal"],
            maxneighbor=c["maxneighbor"],
            random_state=0,
        )
        model.fit(X)
        t1 = time.time()

        cost = calculate_cost(X, model.medoid_indices_)
        print(
            f"numlocal={c['numlocal']:2d}  maxneighbor={c['maxneighbor']:4d}  time={t1-t0:.3f}s  cost={cost:.2f}"
        )


if __name__ == "__main__":
    main()
