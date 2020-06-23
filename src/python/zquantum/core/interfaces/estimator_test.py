import unittest
from unittest.mock import patch
from .estimator import Estimator
from ..measurement import ExpectationValues


class TestEstimatorInterface(unittest.TestCase):
    @patch.object(Estimator, "__abstractmethods__", set())
    def test_ignore_parameter(self):
        with self.assertLogs("z-quantum-core", level="WARN") as context_manager:
            estimator = Estimator()
            estimator_name = type(estimator).__name__
            parameter_names = ["x", "y", "z"]
            parameter_values = [1, 2, 3]
            expected_logs = list(
                [
                    (
                        yield "WARNING:z-quantum-core:{} = {}. {} does not use {}. The value was ignored.".format(
                            pn, pv, estimator_name, pn
                        )
                    )
                    for (pn, pv) in zip(parameter_names, parameter_values)
                ]
            )

            for pn, pv in zip(parameter_names, parameter_values):
                estimator._log_ignore_parameter(estimator_name, pn, pv)
            self.assertEqual(context_manager.output, expected_logs)


class EstimatorTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
    # self.estimators
    # self.backend
    # self.circuit
    # self.operator
    # self.n_samples
    # self.epsilon
    # self.delta

    def test_get_estimated_expectation_values_returns_expectation_values(self):
        for estimator in self.estimators:
            # Given
            # When
            value = estimator.get_estimated_expectation_values(
                self.backend,
                self.circuit,
                self.operator,
                n_samples=self.n_samples,
                epsilon=self.epsilon,
                delta=self.delta,
            )
            # Then
            self.assertIsInstance(value, ExpectationValues)

    def test_get_estimated_expectation_values(self):
        for estimator in self.estimators:
            # Given
            # When
            values = estimator.get_estimated_expectation_values(
                self.backend,
                self.circuit,
                self.operator,
                n_samples=self.n_samples,
                epsilon=self.epsilon,
                delta=self.delta,
            ).values
            value = values[0]
            # Then
            self.assertTrue(len(values) == 1)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)


def parameter_is_ignored(
    self, estimator, estimator_name, parameter_name, parameter_value
):
    with self.assertLogs("z-quantum-core", level="WARN") as context_manager:
        # Given
        expected_log = "WARNING:z-quantum-core:{} = {}. {} does not use {}. The value was ignored.".format(
            parameter_name, parameter_value, estimator_name, parameter_name
        )
        # When
        if parameter_name == "n_samples":
            self.n_samples = parameter_value
        if parameter_name == "epsilon":
            self.epsilon = parameter_value
        if parameter_name == "delta":
            self.delta = parameter_value

        values = estimator.get_estimated_expectation_values(
            self.backend,
            self.circuit,
            self.operator,
            n_samples=self.n_samples,
            epsilon=self.epsilon,
            delta=self.delta,
        )
        # Then
        self.assertIn(expected_log, context_manager.output)
