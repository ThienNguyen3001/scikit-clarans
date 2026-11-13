"""Benchmarks for prediction, transformation, and inertia tracking."""

import numpy as np
from clarans import FastCLARANS


class Inference:
    """Benchmark predict, transform, and cluster inertia tracking."""

    def setup(self) -> None:
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(1000, 10)
        self.X_test = rng.randn(300, 10)

        self.model = FastCLARANS(n_clusters=5, num_local=2, random_state=42)
        self.model.fit(self.X_train)

    def time_predict(self) -> None:
        self.model.predict(self.X_test)

    def time_transform(self) -> None:
        self.model.transform(self.X_test)

    def track_inertia(self) -> float:
        return float(self.model.inertia_)
