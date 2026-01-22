import warnings

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

from clarans import CLARANS

X, _ = make_blobs(n_samples=500, centers=4, n_features=10, random_state=42)


def clustering_silhouette_scorer(estimator, X):
    labels = estimator.predict(X)
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)


param_grid = {
    "n_clusters": [3, 4, 5],
    "numlocal": [2, 5, 10],
    "init": ["k-medoids++", "random", "heuristic"],
    "maxneighbor": [None, 50],
}


grid_search = GridSearchCV(
    estimator=CLARANS(random_state=42),
    param_grid=param_grid,
    scoring=clustering_silhouette_scorer,
    cv=3,
    verbose=1,
    n_jobs=-1,
)

print("Starting grid search (GridSearchCV)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grid_search.fit(X)

print("\n" + "=" * 50)
print("GRID SEARCH RESULTS")
print("=" * 50)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best mean silhouette score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
print("\nBest model information:")
print(f"  - Number of medoids found: {len(best_model.medoid_indices_)}")
print(f"  - Medoid indices: {best_model.medoid_indices_}")

print("\nTop 3 best configurations:")
results = grid_search.cv_results_
indices = np.argsort(results["mean_test_score"])[::-1][:3]
for i in indices:
    print(
        f"  Rank {results['rank_test_score'][i]}: "
        f"Score={results['mean_test_score'][i]:.4f} | "
        f"Params={results['params'][i]}"
    )
