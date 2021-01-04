import numpy as np
import pytest
from pyquil import Program
from pyquil.gates import X, CNOT, H
from pyquil.wavefunction import Wavefunction
from openfermion import QubitOperator, IsingOperator

from ..circuit import Circuit
from ..measurement import Measurements, ExpectationValues
from ..bitstring_distribution import BitstringDistribution


class QuantumBackendTests:
    def test_run_circuit_and_measure_correct_indexing(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        circuit = Circuit(Program(X(0), X(0), X(1), X(1), X(2)))
        n_samples = 100
        # When
        backend.n_samples = n_samples
        measurements = backend.run_circuit_and_measure(circuit)

        # Then (since SPAM error could result in unexpected bitstrings, we make sure the most common bitstring is
        #   the one we expect)
        counts = measurements.get_counts()
        assert max(counts, key=counts.get) == "001"

    @pytest.mark.parametrize("n_shots", [1, 2, 10, 100])
    def test_run_circuit_and_measure_correct_num_measurements(self, backend, n_shots):
        # Given
        circuit = Circuit(Program(X(0), X(0), X(1), X(1), X(2)))

        # When
        backend.n_samples = n_shots
        measurements = backend.run_circuit_and_measure(circuit)

        # Then
        assert isinstance(measurements, Measurements)
        assert len(measurements.bitstrings) == n_shots

    def test_if_all_measurements_have_the_same_number_of_bits(self, backend):
        # Given
        circuit = Circuit(Program(X(0), X(0), X(1), X(1), X(2)))

        # When
        backend.n_samples = 100
        measurements = backend.run_circuit_and_measure(circuit)

        # Then
        assert all(len(bitstring) == 3 for bitstring in measurements.bitstrings)

    def test_get_expectation_values_identity(self, backend):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        operator = IsingOperator("[]")
        target_expectation_values = np.array([1])
        n_samples = 1
        # When
        backend.n_samples = 1
        expectation_values = backend.get_expectation_values(circuit, operator)
        # Then
        assert isinstance(expectation_values, ExpectationValues)
        assert isinstance(expectation_values.values, np.ndarray)
        assert expectation_values.values == pytest.approx(
            target_expectation_values, abs=1e-15
        )

    def test_get_expectation_values_empty_op(self, backend):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        operator = IsingOperator()
        # When
        backend.n_samples = 1
        expectation_values = backend.get_expectation_values(circuit, operator)
        # Then
        assert expectation_values.values == pytest.approx(0.0, abs=1e-7)

    def test_get_expectation_values_for_circuitset(self, backend):
        # Given
        num_circuits = 10
        circuitset = [
            Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2))) for _ in range(num_circuits)
        ]
        operator = IsingOperator("[]")
        target_expectation_values = np.array([1])

        # When
        backend.n_samples = 1
        expectation_values_set = backend.get_expectation_values_for_circuitset(
            circuitset, operator
        )

        # Then
        assert len(expectation_values_set) == num_circuits

        for expectation_values in expectation_values_set:
            assert isinstance(expectation_values, ExpectationValues)
            assert isinstance(expectation_values.values, np.ndarray)
            assert expectation_values.values == pytest.approx(
                target_expectation_values, abs=1e-15
            )

    def test_get_bitstring_distribution(self, backend):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        n_samples = 1000

        backend.n_samples = n_samples

        # When
        bitstring_distribution = backend.get_bitstring_distribution(circuit)

        # Then
        assert isinstance(bitstring_distribution, BitstringDistribution)
        assert bitstring_distribution.get_qubits_number() == 3
        assert bitstring_distribution.distribution_dict["000"] > 1 / 3
        assert bitstring_distribution.distribution_dict["111"] > 1 / 3


class QuantumSimulatorTests(QuantumBackendTests):
    def test_get_wavefunction(self, wf_simulator):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert isinstance(wavefunction, Wavefunction)
        assert len(wavefunction.probabilities()) == 8
        assert wavefunction[0] == pytest.approx((1 / np.sqrt(2) + 0j), abs=1e-7)
        assert wavefunction[7] == pytest.approx((1 / np.sqrt(2) + 0j), abs=1e-7)

    def test_get_exact_expectation_values(self, wf_simulator):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        qubit_operator = QubitOperator("2[] - [Z0 Z1] + [X0 X2]")
        target_expectation_values = np.array([2.0, -1.0, 0.0])

        # When
        expectation_values = wf_simulator.get_exact_expectation_values(
            circuit, qubit_operator
        )
        # Then
        assert expectation_values.values == pytest.approx(
            target_expectation_values, abs=1e-15
        )
        assert isinstance(expectation_values.values, np.ndarray)

    def test_get_exact_expectation_values_empty_op(self, wf_simulator):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        qubit_operator = QubitOperator()
        target_value = 0.0
        # When
        expectation_values = wf_simulator.get_exact_expectation_values(
            circuit, qubit_operator
        )
        # Then
        assert sum(expectation_values.values) == pytest.approx(target_value, abs=1e-7)

    def test_get_bitstring_distribution_wf_simulators(self, wf_simulator):
        # Given
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))

        # When
        bitstring_distribution = wf_simulator.get_bitstring_distribution(circuit)

        # Then
        assert isinstance(bitstring_distribution, BitstringDistribution)
        assert bitstring_distribution.get_qubits_number() == 3
        assert bitstring_distribution.distribution_dict["000"] == pytest.approx(
            0.5, abs=1e-7
        )
        assert bitstring_distribution.distribution_dict["111"] == pytest.approx(
            0.5, abs=1e-7
        )
