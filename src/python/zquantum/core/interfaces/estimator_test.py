"""Test case prototypes that can be used in other projects.

Note that this file won't be executed on its own by pytest.
You need to define your own test cases that inherit from the ones defined here.
"""


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
