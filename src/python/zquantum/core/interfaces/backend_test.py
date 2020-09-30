import unittest
import numpy as np
from pyquil import Program
from pyquil.gates import X, CNOT, H
from pyquil.wavefunction import Wavefunction
from openfermion import QubitOperator, IsingOperator

from ..circuit import Circuit
from ..measurement import Measurements, ExpectationValues
from ..bitstring_distribution import BitstringDistribution


class QuantumBackendTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
    # self.backends

    def test_run_circuit_and_measure_correct_indexing(self):
        # Note: this test may fail with noisy devices
        # Given
        circuit = Circuit(Program(X(0), X(0), X(1), X(1), X(2)))
        n_samples = 100
        # When
        for backend in self.backends:
            backend.n_samples = n_samples
            measurements = backend.run_circuit_and_measure(circuit)

            # Then (since SPAM error could result in unexpected bitstrings, we make sure the most common bitstring is
            #   the one we expect)
            counts = measurements.get_counts()
            self.assertEqual(max(counts, key=counts.get), "001")

    def test_run_circuit_and_measure_correct_num_measurements(self):
        # Given
        circuit = Circuit(Program(X(0), X(0), X(1), X(1), X(2)))
        n_samples = [1, 2, 10, 100]
        # When
        for n_shots in n_samples:

            for backend in self.backends:
                backend.n_samples = n_shots
                measurements = backend.run_circuit_and_measure(circuit)

                # Then
                self.assertIsInstance(measurements, Measurements)
                self.assertEqual(len(measurements.bitstrings), n_shots)

                for bitstring in measurements.bitstrings:
                    self.assertEqual(len(bitstring), 3)

    def test_get_expectation_values_identity(self):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        operator = IsingOperator("[]")
        target_expectation_values = np.array([1])
        n_samples = 1
        # When
        for backend in self.backends:
            backend.n_samples = 1
            expectation_values = backend.get_expectation_values(circuit, operator)
            # Then
            self.assertIsInstance(expectation_values, ExpectationValues)
            self.assertIsInstance(expectation_values.values, np.ndarray)
            np.testing.assert_array_almost_equal(
                expectation_values.values, target_expectation_values, decimal=15
            )

    def test_get_expectation_values_empty_op(self):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        operator = IsingOperator()
        # When
        for backend in self.backends:
            backend.n_samples = 1
            expectation_values = backend.get_expectation_values(circuit, operator)
            # Then
            self.assertAlmostEqual(sum(expectation_values.values), 0.0)

    def test_get_expectation_values_for_circuitset(self):
        # Given
        num_circuits = 10
        circuitset = [
            Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2))) for _ in range(num_circuits)
        ]
        operator = IsingOperator("[]")
        target_expectation_values = np.array([1])

        # When
        for backend in self.backends:
            backend.n_samples = 1
            expectation_values_set = backend.get_expectation_values_for_circuitset(
                circuitset, operator
            )

            # Then
            self.assertEqual(len(expectation_values_set), num_circuits)

            for expectation_values in expectation_values_set:
                self.assertIsInstance(expectation_values, ExpectationValues)
                self.assertIsInstance(expectation_values.values, np.ndarray)
                np.testing.assert_array_almost_equal(
                    expectation_values.values, target_expectation_values, decimal=15
                )

    def test_get_bitstring_distribution(self):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        n_samples = 1000

        for backend in self.backends:
            backend.n_samples = n_samples

            # When
            bitstring_distribution = backend.get_bitstring_distribution(circuit)

            # Then
            self.assertEqual(type(bitstring_distribution), BitstringDistribution)
            self.assertEqual(bitstring_distribution.get_qubits_number(), 3)
            self.assertGreater(bitstring_distribution.distribution_dict["000"], 1 / 3)
            self.assertGreater(bitstring_distribution.distribution_dict["111"], 1 / 3)


class QuantumSimulatorTests(QuantumBackendTests):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
    # self.backends
    # self.wf_simulators

    def test_get_wavefunction(self):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))

        # When
        for simulator in self.wf_simulators:
            wavefunction = simulator.get_wavefunction(circuit)

            # Then
            self.assertIsInstance(wavefunction, Wavefunction)
            self.assertEqual(len(wavefunction.probabilities()), 8)
            self.assertAlmostEqual(wavefunction[0], (1 / np.sqrt(2) + 0j))
            self.assertAlmostEqual(wavefunction[7], (1 / np.sqrt(2) + 0j))

    def test_get_exact_expectation_values(self):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        qubit_operator = QubitOperator("2[] - [Z0 Z1] + [X0 X2]")
        target_values = np.array([2.0, -1.0, 0.0])

        # When
        for simulator in self.wf_simulators:
            expectation_values = simulator.get_exact_expectation_values(
                circuit, qubit_operator
            )
            # Then
            np.testing.assert_array_almost_equal(
                expectation_values.values, target_values
            )
            self.assertIsInstance(expectation_values.values, np.ndarray)

    def test_get_exact_expectation_values_empty_op(self):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        qubit_operator = QubitOperator()
        target_value = 0.0
        # When
        for simulator in self.wf_simulators:
            expectation_values = simulator.get_exact_expectation_values(
                circuit, qubit_operator
            )
            # Then
            self.assertAlmostEqual(sum(expectation_values.values), target_value)

    def test_get_bitstring_distribution_wf_simulators(self):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))

        for wf_simulator in self.wf_simulators:
            # When
            bitstring_distribution = wf_simulator.get_bitstring_distribution(circuit)

            # Then
            self.assertEqual(type(bitstring_distribution), BitstringDistribution)
            self.assertEqual(bitstring_distribution.get_qubits_number(), 3)
            self.assertAlmostEqual(
                bitstring_distribution.distribution_dict["000"], 1 / 2
            )
            self.assertAlmostEqual(
                bitstring_distribution.distribution_dict["111"], 1 / 2
            )

