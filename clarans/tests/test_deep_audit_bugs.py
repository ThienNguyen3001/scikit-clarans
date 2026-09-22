"""Tests demonstrating and verifying new bugs discovered in scikit-clarans deep audit.

1. Direction inversion for asymmetric callable metrics in _compute_1_vs_n.
2. Handling of infinities (+inf, -inf) in Cython is_matrix_symmetric.
3. NaN recovery for second-nearest medoid in Cython update_cache_2min.
4. Input validation for medoid indices in calculate_cost.
"""

import numpy as np
import pytest

from clarans import CLARANS, FastCLARANS, calculate_cost
from clarans.utils import HAS_CYTHON

if HAS_CYTHON:
    from clarans._core import is_matrix_symmetric, update_cache_2min


class TestDeepAuditBugs:
    def test_asymmetric_callable_metric_inertia_consistency(self):
        """CLARANS and FastCLARANS must compute candidate distances as d(sample -> candidate).
        
        When using an asymmetric callable metric, model.inertia_ must strictly equal
        the actual clustering cost sum(min(transform(X), axis=1)) and -score(X).
        """
        def asymmetric_metric(x, y):
            diff = x[0] - y[0]
            return diff * 10.0 if diff > 0 else -diff * 1.0

        rng = np.random.RandomState(42)
        X = rng.uniform(0, 100, (30, 1))

        for ModelClass in [CLARANS, FastCLARANS]:
            model = ModelClass(
                n_clusters=3,
                metric=asymmetric_metric,
                max_neighbors=50,
                num_local=2,
                random_state=42,
            )
            model.fit(X)

            # Expected inertia is the sum of distances from each point in X to its nearest medoid
            dists = model.transform(X)
            expected_inertia = float(np.sum(np.min(dists, axis=1)))

            # Check that model.inertia_ matches true cost
            assert np.isclose(
                model.inertia_, expected_inertia, rtol=1e-5, atol=1e-8
            ), (
                f"{ModelClass.__name__} reported inertia_={model.inertia_} "
                f"but transform sum is {expected_inertia}"
            )

            # Check that score(X) matches -model.inertia_
            assert np.isclose(
                model.score(X), -model.inertia_, rtol=1e-5, atol=1e-8
            ), (
                f"{ModelClass.__name__} score(X)={model.score(X)} "
                f"does not match -inertia_={-model.inertia_}"
            )

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython core required")
    def test_is_matrix_symmetric_infinities(self):
        """is_matrix_symmetric must handle +inf and -inf correctly according to IEEE 754."""
        # 1. Opposite signed infinities (+inf vs -inf): MUST be False
        D_opp_inf = np.array([[0.0, np.inf], [-np.inf, 0.0]], dtype=np.float64)
        assert not is_matrix_symmetric(D_opp_inf, 2), (
            "Matrix with (+inf, -inf) should be detected as asymmetric!"
        )

        # 2. Same signed infinities (+inf vs +inf): MUST be True
        D_same_inf = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
        assert is_matrix_symmetric(D_same_inf, 2), (
            "Matrix with (+inf, +inf) should be detected as symmetric!"
        )

        # 3. +inf vs finite: MUST be False
        D_inf_finite = np.array([[0.0, np.inf], [5.0, 0.0]], dtype=np.float64)
        assert not is_matrix_symmetric(D_inf_finite, 2), (
            "Matrix with (+inf, 5.0) should be detected as asymmetric!"
        )

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython core required")
    def test_update_cache_2min_nan_recovery(self):
        """update_cache_2min must not let a NaN at position 0 permanently poison second_dist."""
        # subD has distance NaN to medoid 0, and finite distances to medoids 1..3
        subD = np.array([[np.nan, 1.0, 3.0, 2.0]], dtype=np.float64)
        near_idx, near_d, second_d = update_cache_2min(subD, 1, 4)

        assert near_idx[0] == 1, f"Nearest medoid should be 1, got {near_idx[0]}"
        assert np.isclose(near_d[0], 1.0), f"Nearest distance should be 1.0, got {near_d[0]}"
        assert not np.isnan(second_d[0]), (
            f"second_dist[0] was poisoned by NaN! Expected 2.0, got {second_d[0]}"
        )
        assert np.isclose(
            second_d[0], 2.0
        ), f"Second nearest distance should be 2.0, got {second_d[0]}"

    def test_calculate_cost_validation(self):
        """calculate_cost must validate medoid_indices using check_medoids."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # 1. Empty medoid indices
        with pytest.raises(ValueError, match="empty"):
            calculate_cost(X, [])

        # 2. Negative indices
        with pytest.raises(ValueError, match="non-negative"):
            calculate_cost(X, [-1])

        # 3. Out-of-bounds indices
        with pytest.raises(ValueError, match="within"):
            calculate_cost(X, [9999])

        # 4. Duplicate indices
        with pytest.raises(ValueError, match="duplicate"):
            calculate_cost(X, [0, 0])
