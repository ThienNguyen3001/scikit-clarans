"""
03_metrics_demo.py
==================
Demonstrate how different distance metrics affect CLARANS clustering.

Run with: python examples/03_metrics_demo.py
"""

import matplotlib.pyplot as plt
from clarans import CLARANS
from clarans.utils import calculate_cost
from sklearn.datasets import make_blobs


def main():
    X, _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=0)
    metrics = ["euclidean", "manhattan", "cosine"]

    costs = []
    models = []
    for metric in metrics:
        model = CLARANS(n_clusters=4, numlocal=3, init="k-medoids++", metric=metric, random_state=42)
        model.fit(X)
        cost = calculate_cost(X, model.medoid_indices_, metric=metric)
        costs.append(cost)
        models.append(model)
        print(f"metric={metric:9s}  cost={cost:.2f}")

    # Plot clustering results for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    for ax, metric, model in zip(axes, metrics, models):
        ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="tab10", s=20)
        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="*", s=150)
        ax.set_title(metric)
    plt.suptitle("Effect of distance metric on CLARANS clustering")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
