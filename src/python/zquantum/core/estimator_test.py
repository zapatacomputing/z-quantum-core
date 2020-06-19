import unittest
from .interfaces.estimator_test import EstimatorTests
from .interfaces.mock_objects import MockQuantumBackend, MockQuantumSimulator
from .estimator import BasicEstimator, ExactEstimator
from .circuit import Circuit
from pyquil import Program
from pyquil.gates import X
from openfermion import QubitOperator


class TestBasicEstimator(unittest.TestCase, EstimatorTests):
    def setUp(self):
        # Setting up inherited tests
        self.operator = QubitOperator("Z0")
        self.circuit = Circuit(Program(X(0)))
        self.backend = MockQuantumSimulator()
        self.estimators = [BasicEstimator()]

    def test_get_estimated_values(self):
        for estimator in self.estimators:
            # Given
            # When
            values = estimator.get_estimated_values(
                self.backend, self.circuit, self.operator
            ).values
            value = values[0]
            # Then
            self.assertTrue(len(values) == 1)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)


class TestExactEstimator(unittest.TestCase, EstimatorTests):
    def setUp(self):
        # Setting up inherited tests
        self.operator = QubitOperator("Z0")
        self.circuit = Circuit(Program(X(0)))
        self.backend = MockQuantumSimulator()
        self.estimators = [ExactEstimator()]

    def test_require_quantum_simulator(self):
        self.backend = MockQuantumBackend()  # Note: setUp() is run before *each* test.
        for estimator in self.estimators:
            with self.assertRaises(AttributeError):
                value = estimator.get_estimated_values(
                    self.backend, self.circuit, self.operator
                )

    def test_get_estimated_values(self):
        for estimator in self.estimators:
            # Given
            # When
            values = estimator.get_estimated_values(
                self.backend, self.circuit, self.operator
            ).values
            value = values[0]
            # Then
            self.assertTrue(len(values) == 1)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
