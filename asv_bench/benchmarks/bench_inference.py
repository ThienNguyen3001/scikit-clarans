"""Benchmarks for prediction, transformation, precomputed inference, and score metrics."""

import numpy as np
from scipy.spatial.distance import cdist
from clarans import FastCLARANS


class DenseInference:
    """Benchmark predict, transform, and score methods across test sample sizes."""

    params = [200, 1000]
    param_names = ["n_test_samples"]

    def setup(self, n_test_samples: int) -> None:
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(800, 10)
        self.X_test = rng.randn(n_test_samples, 10)

        self.model = FastCLARANS(n_clusters=5, num_local=1, random_state=42)
        self.model.fit(self.X_train)

    def time_predict(self, n_test_samples: int) -> None:
        self.model.predict(self.X_test)

    def time_transform(self, n_test_samples: int) -> None:
        self.model.transform(self.X_test)

    def time_score(self, n_test_samples: int) -> None:
        self.model.score(self.X_test)


class PrecomputedInference:
    """Benchmark prediction and transformation on rectangular precomputed distance matrices."""

    def setup(self) -> None:
        rng = np.random.RandomState(42)
        X_train = rng.randn(600, 10)
        X_test = rng.randn(200, 10)

        D_train = cdist(X_train, X_train, metric="euclidean")
        self.D_test = cdist(X_test, X_train, metric="euclidean")

        self.model = FastCLARANS(n_clusters=5, metric="precomputed", num_local=1, random_state=42)
        self.model.fit(D_train)

    def time_predict_precomputed(self) -> None:
        self.model.predict(self.D_test)

    def time_transform_precomputed(self) -> None:
        self.model.transform(self.D_test)


class ClusterQualityTracking:
    """Track numerical cluster inertia and unseen test score invariants across commits."""

    def setup(self) -> None:
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(600, 10)
        self.X_test = rng.randn(200, 10)

        self.model = FastCLARANS(n_clusters=4, num_local=2, random_state=42)
        self.model.fit(self.X_train)

    def track_train_inertia(self) -> float:
        return float(self.model.inertia_)

    def track_test_score(self) -> float:
        return float(self.model.score(self.X_test))
