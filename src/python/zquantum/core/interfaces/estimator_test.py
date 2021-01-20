import pytest
from ..measurement import ExpectationValues


class EstimatorTests:
    def test_get_estimated_expectation_values_returns_expectation_values(
        self, estimator, backend, circuit, operator, n_samples, epsilon, delta
    ):
        value = estimator.get_estimated_expectation_values(
            backend,
            circuit,
            operator,
            n_samples=n_samples,
            epsilon=epsilon,
            delta=delta,
        )
        # Then
        assert type(value) is ExpectationValues
