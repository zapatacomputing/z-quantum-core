import unittest
from .interfaces.estimator_test import EstimatorTests, parameter_is_ignored
from .interfaces.mock_objects import MockQuantumBackend, MockQuantumSimulator
from .estimator import (
    BasicEstimator,
    ExactEstimator,
    get_context_selection_circuit,
)
from .circuit import Circuit
from pyquil import Program
from pyquil.gates import X
from openfermion import QubitOperator, qubit_operator_sparse, IsingOperator
import numpy as np


class TestEstimatorUtils(unittest.TestCase):
    def test_get_context_selection_circuit_offdiagonal(self):
        term = ((0, "X"), (1, "Y"))
        circuit, ising_operator = get_context_selection_circuit(term)

        # Need to convert to QubitOperator in order to get matrix representation
        qubit_operator = QubitOperator()
        for ising_term in ising_operator.terms:
            qubit_operator += QubitOperator(
                ising_term, ising_operator.terms[ising_term]
            )

        target_unitary = qubit_operator_sparse(QubitOperator(term))
        transformed_unitary = (
            circuit.to_unitary().conj().T
            @ qubit_operator_sparse(qubit_operator)
            @ circuit.to_unitary()
        )

        self.assertTrue(np.allclose(target_unitary.todense(), transformed_unitary))

    def test_get_context_selection_circuit_diagonal(self):
        term = ((4, "Z"), (2, "Z"))
        circuit, ising_operator = get_context_selection_circuit(term)
        self.assertEqual(len(circuit.gates), 0)
        self.assertEqual(ising_operator, IsingOperator(term))


class TestBasicEstimator(unittest.TestCase, EstimatorTests):
    def setUp(self):
        # Setting up inherited tests
        self.operator = QubitOperator("Z0")
        self.circuit = Circuit(Program(X(0)))
        self.backend = MockQuantumBackend(n_samples=20)
        self.n_samples = 10
        self.epsilon = None
        self.delta = None
        self.estimators = [BasicEstimator()]

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
            self.assertGreaterEqual(value, -1)
            self.assertLessEqual(value, 1)

    def test_get_estimated_expectation_values_samples_from_backend(self):
        for estimator in self.estimators:
            # Given
            # When
            values = estimator.get_estimated_expectation_values(
                self.backend,
                self.circuit,
                self.operator,
                epsilon=self.epsilon,
                delta=self.delta,
            ).values
            value = values[0]
            # Then
            self.assertTrue(len(values) == 1)
            self.assertGreaterEqual(value, -1)
            self.assertLessEqual(value, 1)

    def test_n_samples_is_restored(self):
        for estimator in self.estimators:
            # Given
            self.backend.n_samples = 5
            # When
            values = estimator.get_estimated_expectation_values(
                self.backend, self.circuit, self.operator, n_samples=10
            )
            # Then
            self.assertEqual(self.backend.n_samples, 5)

    def test_epsilon_is_ignored(self):
        for estimator in self.estimators:
            # Given
            estimator_name = type(estimator).__name__
            parameter_name = "epsilon"
            parameter_value = 0.1
            parameter_is_ignored(
                self, estimator, estimator_name, parameter_name, parameter_value
            )

    def test_delta_is_ignored(self):
        for estimator in self.estimators:
            # Given
            estimator_name = type(estimator).__name__
            parameter_name = "delta"
            parameter_value = 0.9
            parameter_is_ignored(
                self, estimator, estimator_name, parameter_name, parameter_value
            )


class TestExactEstimator(unittest.TestCase, EstimatorTests):
    def setUp(self):
        # Setting up inherited tests
        self.operator = QubitOperator("Z0")
        self.circuit = Circuit(Program(X(0)))
        self.backend = MockQuantumSimulator()
        self.n_samples = None
        self.epsilon = None
        self.delta = None
        self.estimators = [ExactEstimator()]

    def test_require_quantum_simulator(self):
        self.backend = MockQuantumBackend()  # Note: setUp() is run before *each* test.
        for estimator in self.estimators:
            with self.assertRaises(AttributeError):
                value = estimator.get_estimated_expectation_values(
                    self.backend, self.circuit, self.operator,
                ).values

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
            self.assertGreaterEqual(value, -1)
            self.assertLessEqual(value, 1)

    def test_n_samples_is_ignored(self):
        for estimator in self.estimators:
            # Given
            estimator_name = type(estimator).__name__
            parameter_name = "n_samples"
            parameter_value = 1
            parameter_is_ignored(
                self, estimator, estimator_name, parameter_name, parameter_value
            )

    def test_epsilon_is_ignored(self):
        for estimator in self.estimators:
            # Given
            estimator_name = type(estimator).__name__
            parameter_name = "epsilon"
            parameter_value = 0.1
            parameter_is_ignored(
                self, estimator, estimator_name, parameter_name, parameter_value
            )

    def test_delta_is_ignored(self):
        for estimator in self.estimators:
            # Given
            estimator_name = type(estimator).__name__
            parameter_name = "delta"
            parameter_value = 0.9
            parameter_is_ignored(
                self, estimator, estimator_name, parameter_name, parameter_value
            )
