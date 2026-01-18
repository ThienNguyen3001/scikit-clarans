"""
06_predict_new_data.py
======================
Fit CLARANS on training data and demonstrate predicting cluster labels for new points.

Run: python examples/06_predict_new_data.py
"""

from clarans import CLARANS
from sklearn.datasets import make_blobs
import numpy as np


def main():
    X, _ = make_blobs(n_samples=450, centers=3, n_features=2, random_state=42)

    model = CLARANS(n_clusters=3, init="k-medoids++", random_state=0)
    model.fit(X)

    # Create a grid of new points
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 60),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 60),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    labels = model.predict(grid)
    print("Predicted labels for grid (shape):", labels.shape)


if __name__ == "__main__":
    main()
