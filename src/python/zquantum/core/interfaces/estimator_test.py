import unittest
from ..measurement import ExpectationValues


class EstimatorTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
    # self.estimators
    # self.backend
    # self.circuit
    # self.operator

    def test_get_estimated_values_returns_expectation_values(self):
        for estimator in self.estimators:
            # Given
            # When
            value = estimator.get_estimated_values(
                self.backend, self.circuit, self.operator
            )
            # Then
            self.assertIsInstance(value, ExpectationValues)
