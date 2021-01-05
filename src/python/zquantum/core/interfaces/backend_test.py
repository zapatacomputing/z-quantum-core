import numpy as np
import pytest
from pyquil import Program
from pyquil.gates import X, CNOT, H
from pyquil.wavefunction import Wavefunction
from openfermion import QubitOperator, IsingOperator

from ..circuit import Circuit, Qubit, Gate
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

    @pytest.mark.parametrize(
        "initial_gate,gate_name,target_amplitudes",
        [
            ["I", "I", [1.0, 0.0]],
            ["H", "I", [1 / np.sqrt(2), 1 / np.sqrt(2)]],
            ["X", "I", [0.0, 1.0]],
            ["I", "X", [0.0, 1.0]],
            ["H", "X", [1 / np.sqrt(2), 1 / np.sqrt(2)]],
            ["X", "X", [1.0, 0.0]],
            ["I", "Y", [0.0, 1.0j]],
            ["H", "Y", [-1j / np.sqrt(2), 1j / np.sqrt(2)]],
            ["X", "Y", [-1.0j, 0.0]],
            ["I", "Z", [1.0, 0.0]],
            ["H", "Z", [1 / np.sqrt(2), -1 / np.sqrt(2)]],
            ["X", "Z", [0.0, -1.0]],
            ["I", "H", [1 / np.sqrt(2), 1 / np.sqrt(2)]],
            ["H", "H", [1.0, 0.0]],
            ["X", "H", [1 / np.sqrt(2), -1 / np.sqrt(2)]],
            ["I", "S", [1.0, 0.0]],
            ["H", "S", [1 / np.sqrt(2), 1j / np.sqrt(2)]],
            ["X", "S", [0.0, 1.0j]],
            ["I", "T", [1.0, 0.0]],
            ["H", "T", [1 / np.sqrt(2), 0.5 + 0.5j]],
            ["X", "T", [0.0, 1 / np.sqrt(2) + 1j / np.sqrt(2)]],
        ],
    )
    def test_1_qubit_non_parametric_gates(
        self, wf_simulator, initial_gate, gate_name, target_amplitudes
    ):
        # Given
        qubit_list = [Qubit(0)]
        gate_1 = Gate(initial_gate, qubits=qubit_list)
        gate_2 = Gate(gate_name, qubits=qubit_list)

        circuit = Circuit()
        circuit.gates = [gate_1, gate_2]

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gate,gate_name,params,target_amplitudes",
        [
            ["I", "Rx", [-np.pi / 2], [np.sqrt(2) / 2, 0.5 * np.sqrt(2) * 1.0j]],
            ["I", "Rx", [0], [1, 0]],
            [
                "I",
                "Rx",
                [np.pi / 5],
                [np.sqrt(np.sqrt(5) / 8 + 5 / 8), -1.0j * (-1 / 4 + np.sqrt(5) / 4)],
            ],
            ["I", "Rx", [np.pi / 2], [np.sqrt(2) / 2, -0.5 * np.sqrt(2) * 1.0j]],
            ["I", "Rx", [np.pi], [0, -1.0j]],
            ["H", "Rx", [-np.pi / 2], [1 / 2 + 0.5 * 1.0j, 1 / 2 + 0.5 * 1.0j]],
            ["H", "Rx", [0], [np.sqrt(2) / 2, np.sqrt(2) / 2]],
            [
                "H",
                "Rx",
                [np.pi / 5],
                [
                    np.sqrt(2) * np.sqrt(np.sqrt(5) / 8 + 5 / 8) / 2
                    - 0.5 * np.sqrt(2) * 1.0j * (-1 / 4 + np.sqrt(5) / 4),
                    np.sqrt(2) * np.sqrt(np.sqrt(5) / 8 + 5 / 8) / 2
                    - 0.5 * np.sqrt(2) * 1.0j * (-1 / 4 + np.sqrt(5) / 4),
                ],
            ],
            ["H", "Rx", [np.pi / 2], [1 / 2 - 0.5 * 1.0j, 1 / 2 - 0.5 * 1.0j]],
            ["H", "Rx", [np.pi], [-0.5 * np.sqrt(2) * 1.0j, -0.5 * np.sqrt(2) * 1.0j]],
            ["X", "Rx", [-np.pi / 2], [0.5 * np.sqrt(2) * 1.0j, np.sqrt(2) / 2]],
            ["X", "Rx", [0], [0, 1]],
            [
                "X",
                "Rx",
                [np.pi / 5],
                [-1.0j * (-1 / 4 + np.sqrt(5) / 4), np.sqrt(np.sqrt(5) / 8 + 5 / 8)],
            ],
            ["X", "Rx", [np.pi / 2], [-0.5 * np.sqrt(2) * 1.0j, np.sqrt(2) / 2]],
            ["X", "Rx", [np.pi], [-1.0j, 0]],
            ["I", "Ry", [-np.pi / 2], [np.sqrt(2) / 2, -np.sqrt(2) / 2]],
            ["I", "Ry", [0], [1, 0]],
            [
                "I",
                "Ry",
                [np.pi / 5],
                [np.sqrt(np.sqrt(5) / 8 + 5 / 8), -1 / 4 + np.sqrt(5) / 4],
            ],
            ["I", "Ry", [np.pi / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]],
            ["I", "Ry", [np.pi], [0, 1]],
            ["H", "Ry", [-np.pi / 2], [1, 0]],
            ["H", "Ry", [0], [np.sqrt(2) / 2, np.sqrt(2) / 2]],
            [
                "H",
                "Ry",
                [np.pi / 5],
                [
                    -np.sqrt(2) * (-1 / 4 + np.sqrt(5) / 4) / 2
                    + np.sqrt(2) * np.sqrt(np.sqrt(5) / 8 + 5 / 8) / 2,
                    np.sqrt(2) * (-1 / 4 + np.sqrt(5) / 4) / 2
                    + np.sqrt(2) * np.sqrt(np.sqrt(5) / 8 + 5 / 8) / 2,
                ],
            ],
            ["H", "Ry", [np.pi / 2], [0, 1]],
            ["H", "Ry", [np.pi], [-np.sqrt(2) / 2, np.sqrt(2) / 2]],
            ["X", "Ry", [-np.pi / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]],
            ["X", "Ry", [0], [0, 1]],
            [
                "X",
                "Ry",
                [np.pi / 5],
                [1 / 4 - np.sqrt(5) / 4, np.sqrt(np.sqrt(5) / 8 + 5 / 8)],
            ],
            ["X", "Ry", [np.pi / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]],
            ["X", "Ry", [np.pi], [-1, 0]],
            ["I", "Rz", [-np.pi / 2], [np.sqrt(2) / 2 + 0.5 * np.sqrt(2) * 1.0j, 0]],
            ["I", "Rz", [0], [1, 0]],
            [
                "I",
                "Rz",
                [np.pi / 5],
                [np.sqrt(np.sqrt(5) / 8 + 5 / 8) - 1.0j * (-1 / 4 + np.sqrt(5) / 4), 0],
            ],
            ["I", "Rz", [np.pi / 2], [np.sqrt(2) / 2 - 0.5 * np.sqrt(2) * 1.0j, 0]],
            ["I", "Rz", [np.pi], [-1.0j, 0]],
            [
                "H",
                "Rz",
                [-np.pi / 2],
                [
                    np.sqrt(2) * (np.sqrt(2) / 2 + 0.5 * np.sqrt(2) * 1.0j) / 2,
                    np.sqrt(2) * (np.sqrt(2) / 2 - 0.5 * np.sqrt(2) * 1.0j) / 2,
                ],
            ],
            ["H", "Rz", [0], [np.sqrt(2) / 2, np.sqrt(2) / 2]],
            [
                "H",
                "Rz",
                [np.pi / 5],
                [
                    np.sqrt(2)
                    * (
                        np.sqrt(np.sqrt(5) / 8 + 5 / 8)
                        - 1.0j * (-1 / 4 + np.sqrt(5) / 4)
                    )
                    / 2,
                    np.sqrt(2)
                    * (
                        np.sqrt(np.sqrt(5) / 8 + 5 / 8)
                        + 1.0j * (-1 / 4 + np.sqrt(5) / 4)
                    )
                    / 2,
                ],
            ],
            [
                "H",
                "Rz",
                [np.pi / 2],
                [
                    np.sqrt(2) * (np.sqrt(2) / 2 - 0.5 * np.sqrt(2) * 1.0j) / 2,
                    np.sqrt(2) * (np.sqrt(2) / 2 + 0.5 * np.sqrt(2) * 1.0j) / 2,
                ],
            ],
            ["H", "Rz", [np.pi], [-0.5 * np.sqrt(2) * 1.0j, 0.5 * np.sqrt(2) * 1.0j]],
            ["X", "Rz", [-np.pi / 2], [0, np.sqrt(2) / 2 - 0.5 * np.sqrt(2) * 1.0j]],
            ["X", "Rz", [0], [0, 1]],
            [
                "X",
                "Rz",
                [np.pi / 5],
                [0, np.sqrt(np.sqrt(5) / 8 + 5 / 8) + 1.0j * (-1 / 4 + np.sqrt(5) / 4)],
            ],
            ["X", "Rz", [np.pi / 2], [0, np.sqrt(2) / 2 + 0.5 * np.sqrt(2) * 1.0j]],
            ["X", "Rz", [np.pi], [0, 1.0j]],
            ["I", "PHASE", [-np.pi / 2], [1, 0]],
            ["I", "PHASE", [0], [1, 0]],
            ["I", "PHASE", [np.pi / 5], [1, 0]],
            ["I", "PHASE", [np.pi / 2], [1, 0]],
            ["I", "PHASE", [np.pi], [1, 0]],
            ["H", "PHASE", [-np.pi / 2], [np.sqrt(2) / 2, -0.5 * np.sqrt(2) * 1.0j]],
            ["H", "PHASE", [0], [np.sqrt(2) / 2, np.sqrt(2) / 2]],
            [
                "H",
                "PHASE",
                [np.pi / 5],
                [
                    np.sqrt(2) / 2,
                    np.sqrt(2)
                    * (1 / 4 + np.sqrt(5) / 4 + 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8))
                    / 2,
                ],
            ],
            ["H", "PHASE", [np.pi / 2], [np.sqrt(2) / 2, 0.5 * np.sqrt(2) * 1.0j]],
            ["H", "PHASE", [np.pi], [np.sqrt(2) / 2, -np.sqrt(2) / 2]],
            ["X", "PHASE", [-np.pi / 2], [0, -1.0j]],
            ["X", "PHASE", [0], [0, 1]],
            [
                "X",
                "PHASE",
                [np.pi / 5],
                [0, 1 / 4 + np.sqrt(5) / 4 + 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8)],
            ],
            ["X", "PHASE", [np.pi / 2], [0, 1.0j]],
            ["X", "PHASE", [np.pi], [0, -1]],
            ["I", "RH", [-np.pi / 2], [np.sqrt(2) / 2 + 0.5 * 1.0j, 0.5 * 1.0j]],
            ["I", "RH", [0], [1, 0]],
            [
                "I",
                "RH",
                [np.pi / 5],
                [
                    np.sqrt(np.sqrt(5) / 8 + 5 / 8)
                    - 0.5 * np.sqrt(2) * 1.0j * (-1 / 4 + np.sqrt(5) / 4),
                    -0.5 * np.sqrt(2) * 1.0j * (-1 / 4 + np.sqrt(5) / 4),
                ],
            ],
            ["I", "RH", [np.pi / 2], [np.sqrt(2) / 2 - 0.5 * 1.0j, -0.5 * 1.0j]],
            ["I", "RH", [np.pi], [-0.5 * np.sqrt(2) * 1.0j, -0.5 * np.sqrt(2) * 1.0j]],
            [
                "H",
                "RH",
                [-np.pi / 2],
                [
                    0.25 * np.sqrt(2) * 1.0j
                    + np.sqrt(2) * (np.sqrt(2) / 2 + 0.5 * 1.0j) / 2,
                    np.sqrt(2) * (np.sqrt(2) / 2 - 0.5 * 1.0j) / 2
                    + 0.25 * np.sqrt(2) * 1.0j,
                ],
            ],
            ["H", "RH", [0], [np.sqrt(2) / 2, np.sqrt(2) / 2]],
            [
                "H",
                "RH",
                [np.pi / 5],
                [
                    -0.5 * 1.0j * (-1 / 4 + np.sqrt(5) / 4)
                    + np.sqrt(2)
                    * (
                        np.sqrt(np.sqrt(5) / 8 + 5 / 8)
                        - 0.5 * np.sqrt(2) * 1.0j * (-1 / 4 + np.sqrt(5) / 4)
                    )
                    / 2,
                    -0.5 * 1.0j * (-1 / 4 + np.sqrt(5) / 4)
                    + np.sqrt(2)
                    * (
                        np.sqrt(np.sqrt(5) / 8 + 5 / 8)
                        + 0.5 * np.sqrt(2) * 1.0j * (-1 / 4 + np.sqrt(5) / 4)
                    )
                    / 2,
                ],
            ],
            [
                "H",
                "RH",
                [np.pi / 2],
                [
                    -0.25 * np.sqrt(2) * 1.0j
                    + np.sqrt(2) * (np.sqrt(2) / 2 - 0.5 * 1.0j) / 2,
                    -0.25 * np.sqrt(2) * 1.0j
                    + np.sqrt(2) * (np.sqrt(2) / 2 + 0.5 * 1.0j) / 2,
                ],
            ],
            ["H", "RH", [np.pi], [-1.0j, 0]],
            ["X", "RH", [-np.pi / 2], [0.5 * 1.0j, np.sqrt(2) / 2 - 0.5 * 1.0j]],
            ["X", "RH", [0], [0, 1]],
            [
                "X",
                "RH",
                [np.pi / 5],
                [
                    -0.5 * np.sqrt(2) * 1.0j * (-1 / 4 + np.sqrt(5) / 4),
                    np.sqrt(np.sqrt(5) / 8 + 5 / 8)
                    + 0.5 * np.sqrt(2) * 1.0j * (-1 / 4 + np.sqrt(5) / 4),
                ],
            ],
            ["X", "RH", [np.pi / 2], [-0.5 * 1.0j, np.sqrt(2) / 2 + 0.5 * 1.0j]],
            ["X", "RH", [np.pi], [-0.5 * np.sqrt(2) * 1.0j, 0.5 * np.sqrt(2) * 1.0j]],
        ],
    )
    def test_1_qubit_parametric_gates(
        self, wf_simulator, initial_gate, gate_name, params, target_amplitudes
    ):
        # Given
        qubit_list = [Qubit(0)]
        gate_1 = Gate(initial_gate, params=params, qubits=qubit_list)
        gate_2 = Gate(gate_name, params=params, qubits=qubit_list)

        circuit = Circuit()
        circuit.gates = [gate_1, gate_2]

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gates,gate_name,target_amplitudes",
        [
            [["I", "I"], "CNOT", [1.0, 0.0, 0.0, 0.0]],
            [["I", "X"], "CNOT", [0.0, 0.0, 1.0, 0.0]],
            [["X", "I"], "CNOT", [0.0, 0.0, 0.0, 1.0]],
            [["X", "X"], "CNOT", [0.0, 1.0, 0.0, 0.0]],
            [["I", "I"], "SWAP", [1.0, 0.0, 0.0, 0.0]],
            [["I", "X"], "SWAP", [0.0, 1.0, 0.0, 0.0]],
            [["X", "I"], "SWAP", [0.0, 0.0, 1.0, 0.0]],
            [["X", "X"], "SWAP", [0.0, 0.0, 0.0, 1.0]],
            [["I", "I"], "CZ", [1.0, 0.0, 0.0, 0.0]],
            [["I", "X"], "CZ", [0.0, 0.0, 1.0, 0.0]],
            [["X", "I"], "CZ", [0.0, 1.0, 0.0, 0.0]],
            [["X", "X"], "CZ", [0.0, 0.0, 0.0, -1.0]],
        ],
    )
    def test_2_qubit_non_parametric_gates(
        self, wf_simulator, initial_gates, gate_name, target_amplitudes
    ):
        # Given
        qubit_list = [Qubit(0), Qubit(1)]
        gate_1 = Gate(initial_gates[0], qubits=[qubit_list[0]])
        gate_2 = Gate(initial_gates[1], qubits=[qubit_list[1]])
        gate_3 = Gate(gate_name, qubits=qubit_list)

        circuit = Circuit()
        circuit.gates = [gate_1, gate_2, gate_3]

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gates,gate_name,params,target_amplitudes",
        [
            [["I", "I"], "CPHASE", [-np.pi / 2], [1, 0, 0, 0]],
            [["I", "I"], "CPHASE", [np.pi / 5], [1, 0, 0, 0]],
            [["I", "I"], "CPHASE", [np.pi / 2], [1, 0, 0, 0]],
            [["I", "I"], "CPHASE", [np.pi], [1, 0, 0, 0]],
            [["X", "X"], "CPHASE", [-np.pi / 2], [0, 0, 0, -1.0j]],
            [
                ["X", "X"],
                "CPHASE",
                [np.pi / 5],
                [
                    0,
                    0,
                    0,
                    1 / 4 + np.sqrt(5) / 4 + 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8),
                ],
            ],
            [["X", "X"], "CPHASE", [np.pi / 2], [0, 0, 0, 1.0j]],
            [["X", "X"], "CPHASE", [np.pi], [0, 0, 0, -1]],
            [
                ["H", "CNOT"],
                "CPHASE",
                [-np.pi / 2],
                [np.sqrt(2) / 2, 0, 0, -0.5 * np.sqrt(2) * 1.0j],
            ],
            [
                ["H", "CNOT"],
                "CPHASE",
                [np.pi / 5],
                [
                    np.sqrt(2) / 2,
                    0,
                    0,
                    np.sqrt(2)
                    * (1 / 4 + np.sqrt(5) / 4 + 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8))
                    / 2,
                ],
            ],
            [
                ["H", "CNOT"],
                "CPHASE",
                [np.pi / 2],
                [np.sqrt(2) / 2, 0, 0, 0.5 * np.sqrt(2) * 1.0j],
            ],
            [["H", "CNOT"], "CPHASE", [np.pi], [np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2]],
            [["I", "I"], "XX", [-np.pi / 2], [0, 0, 0, -1.0j]],
            [
                ["I", "I"],
                "XX",
                [np.pi / 5],
                [1 / 4 + np.sqrt(5) / 4, 0, 0, 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8)],
            ],
            [["I", "I"], "XX", [np.pi / 2], [0, 0, 0, 1.0j]],
            [["I", "I"], "XX", [np.pi], [-1, 0, 0, 0]],
            [["X", "X"], "XX", [-np.pi / 2], [-1.0j, 0, 0, 0]],
            [
                ["X", "X"],
                "XX",
                [np.pi / 5],
                [1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8), 0, 0, 1 / 4 + np.sqrt(5) / 4],
            ],
            [["X", "X"], "XX", [np.pi / 2], [1.0j, 0, 0, 0]],
            [["X", "X"], "XX", [np.pi], [0, 0, 0, -1]],
            [
                ["H", "CNOT"],
                "XX",
                [-np.pi / 2],
                [-0.5 * np.sqrt(2) * 1.0j, 0, 0, -0.5 * np.sqrt(2) * 1.0j],
            ],
            [
                ["H", "CNOT"],
                "XX",
                [np.pi / 5],
                [
                    np.sqrt(2) * (1 / 4 + np.sqrt(5) / 4) / 2
                    + 0.5 * np.sqrt(2) * 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8),
                    0,
                    0,
                    np.sqrt(2) * (1 / 4 + np.sqrt(5) / 4) / 2
                    + 0.5 * np.sqrt(2) * 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8),
                ],
            ],
            [
                ["H", "CNOT"],
                "XX",
                [np.pi / 2],
                [0.5 * np.sqrt(2) * 1.0j, 0, 0, 0.5 * np.sqrt(2) * 1.0j],
            ],
            [["H", "CNOT"], "XX", [np.pi], [-np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2]],
            [["I", "I"], "YY", [-np.pi / 2], [0, 0, 0, -1.0j]],
            [
                ["I", "I"],
                "YY",
                [np.pi / 5],
                [1 / 4 + np.sqrt(5) / 4, 0, 0, 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8)],
            ],
            [["I", "I"], "YY", [np.pi / 2], [0, 0, 0, 1.0j]],
            [["I", "I"], "YY", [np.pi], [-1, 0, 0, 0]],
            [["X", "X"], "YY", [-np.pi / 2], [-1.0j, 0, 0, 0]],
            [
                ["X", "X"],
                "YY",
                [np.pi / 5],
                [1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8), 0, 0, 1 / 4 + np.sqrt(5) / 4],
            ],
            [["X", "X"], "YY", [np.pi / 2], [1.0j, 0, 0, 0]],
            [["X", "X"], "YY", [np.pi], [0, 0, 0, -1]],
            [
                ["H", "CNOT"],
                "YY",
                [-np.pi / 2],
                [-0.5 * np.sqrt(2) * 1.0j, 0, 0, -0.5 * np.sqrt(2) * 1.0j],
            ],
            [
                ["H", "CNOT"],
                "YY",
                [np.pi / 5],
                [
                    np.sqrt(2) * (1 / 4 + np.sqrt(5) / 4) / 2
                    + 0.5 * np.sqrt(2) * 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8),
                    0,
                    0,
                    np.sqrt(2) * (1 / 4 + np.sqrt(5) / 4) / 2
                    + 0.5 * np.sqrt(2) * 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8),
                ],
            ],
            [
                ["H", "CNOT"],
                "YY",
                [np.pi / 2],
                [0.5 * np.sqrt(2) * 1.0j, 0, 0, 0.5 * np.sqrt(2) * 1.0j],
            ],
            [["H", "CNOT"], "YY", [np.pi], [-np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2]],
            [["I", "I"], "ZZ", [-np.pi / 2], [-1.0j, 0, 0, 0]],
            [
                ["I", "I"],
                "ZZ",
                [np.pi / 5],
                [
                    1 / 4 + np.sqrt(5) / 4 + 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8),
                    0,
                    0,
                    0,
                ],
            ],
            [["I", "I"], "ZZ", [np.pi / 2], [1.0j, 0, 0, 0]],
            [["I", "I"], "ZZ", [np.pi], [-1, 0, 0, 0]],
            [["X", "X"], "ZZ", [-np.pi / 2], [0, 0, 0, -1.0j]],
            [
                ["X", "X"],
                "ZZ",
                [np.pi / 5],
                [
                    0,
                    0,
                    0,
                    1 / 4 + np.sqrt(5) / 4 + 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8),
                ],
            ],
            [["X", "X"], "ZZ", [np.pi / 2], [0, 0, 0, 1.0j]],
            [["X", "X"], "ZZ", [np.pi], [0, 0, 0, -1]],
            [
                ["H", "CNOT"],
                "ZZ",
                [-np.pi / 2],
                [-0.5 * np.sqrt(2) * 1.0j, 0, 0, -0.5 * np.sqrt(2) * 1.0j],
            ],
            [
                ["H", "CNOT"],
                "ZZ",
                [np.pi / 5],
                [
                    np.sqrt(2)
                    * (1 / 4 + np.sqrt(5) / 4 + 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8))
                    / 2,
                    0,
                    0,
                    np.sqrt(2)
                    * (1 / 4 + np.sqrt(5) / 4 + 1.0j * np.sqrt(5 / 8 - np.sqrt(5) / 8))
                    / 2,
                ],
            ],
            [
                ["H", "CNOT"],
                "ZZ",
                [np.pi / 2],
                [0.5 * np.sqrt(2) * 1.0j, 0, 0, 0.5 * np.sqrt(2) * 1.0j],
            ],
            [["H", "CNOT"], "ZZ", [np.pi], [-np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2]],
        ],
    )
    def test_2_qubit_parametric_gates(
        self, wf_simulator, initial_gates, gate_name, params, target_amplitudes
    ):
        # Given
        qubit_list = [Qubit(0), Qubit(1)]
        gate_1 = Gate(initial_gates[0], qubits=[qubit_list[0]])
        if initial_gates[1] == "CNOT":
            gate_2 = Gate(initial_gates[1], qubits=qubit_list)
        else:
            gate_2 = Gate(initial_gates[1], qubits=[qubit_list[1]])
        gate_3 = Gate(gate_name, params=params, qubits=qubit_list)

        circuit = Circuit()

        circuit.gates = [gate_1, gate_2, gate_3]
        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)