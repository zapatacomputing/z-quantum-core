import pytest
from ..measurement import ExpectationValues


class EstimatorTests:
    def test_get_estimated_expectation_values_returns_expectation_values(
        self,
        estimator,
        backend,
        circuit,
        target_operator,
        n_samples,
        **estimator_kwargs
    ):
        value = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=target_operator,
            n_samples=n_samples,
            **estimator_kwargs
        )
        # Then
        assert type(value) is ExpectationValues
