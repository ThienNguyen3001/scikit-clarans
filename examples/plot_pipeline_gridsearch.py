"""
=====================================================
Hyperparameter Tuning with GridSearchCV and Pipelines
=====================================================

This example integrates CLARANS with scikit-learn's standard model selection
tooling, executing a parameter grid search scored via the Silhouette metric.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from clarans import CLARANS


def clustering_silhouette_scorer(estimator, X):
    labels = estimator.predict(X)
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)


def main():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=6, random_state=42)

    param_grid = {
        "n_clusters": [2, 3, 4],
        "num_local": [2, 4],
        "init": ["k-medoids++", "random"],
    }

    grid_search = GridSearchCV(
        estimator=CLARANS(random_state=42),
        param_grid=param_grid,
        scoring=clustering_silhouette_scorer,
        cv=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X)

    print("Best Parameters:", grid_search.best_params_)
    print(f"Best Silhouette Score: {grid_search.best_score_:.4f}")

    # Plot scores for each candidate parameter set
    mean_scores = grid_search.cv_results_["mean_test_score"]
    labels = [
        f"k={p['n_clusters']}, {p['init']}"
        for p in grid_search.cv_results_["params"]
    ]

    plt.figure(figsize=(9, 4.5))
    plt.barh(range(len(mean_scores)), mean_scores, color="#4c72b0")
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.xlabel("Mean Silhouette Score")
    plt.title("GridSearchCV Model Selection for CLARANS")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
