import numpy as np
import pytest
from pyquil import Program
from pyquil.gates import X, CNOT, H
from pyquil.wavefunction import Wavefunction
from openfermion import QubitOperator, IsingOperator

from ..circuit import Circuit, Qubit, Gate
from ..measurement import Measurements, ExpectationValues
from ..bitstring_distribution import BitstringDistribution
from ..estimator import BasicEstimator
from ..testing.test_cases_for_backend_tests import *

"""
Note regarding testing specific gates.

To test that a gate is properly implemented, we can ask for its matrix representation 
and check that each entry is correct. In some quantum simulator packages, 
returning this matrix representation is either not possible or difficult to implement. 
In such cases, we can check that the gate implementation is correct by ensuring that 
the gate transforms input states to output states as expected. If the simulator has 
the capability to provide the wavefunction as an output, then we can check that 
the entries of the transformed wavefunction are correct. If the simulator does not 
have the capability of providing the wavefunction as an output, but only gives 
bitstring samples from the wavefunction, then we can check that the bitstring statistics 
are as expected after taking sufficiently many samples. In both of these cases where 
we cannot directly check the matrix corresponding to the gate, we must check the action 
of the gate on multiple inputs (and outputs in the sampling case). We can picture 
this process as a kind of "quantum process tomography" for gate unit testing. Mathematically, 
correctness is ensured if the span of the input and outputs spans the full vector space. 
Checking a tomographically complete set of input and outputs could be time consuming, 
especially in the case of sampling. Furthermore, we expect that the bugs that will occur 
will lead to an effect on many inputs (rather than, say, a single input-output pair). 
Therefore, we are taking here a slightly lazy, but efficient approach to testing these gates 
by testing how they transform a tomographically incomplete set of input and outputs.

Gates tests use `backend_for_gates_test` instead of `backend` as an input parameter because:
a) it has high chance of failing for noisy backends
b) having execution time in mind it's a good idea to use lower number of samples.
"""


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


class QuantumBackendGatesTests:
    @pytest.mark.parametrize(
        "initial_gate,tested_gate,target_values",
        one_qubit_non_parametric_gates_exp_vals_test_set,
    )
    def test_one_qubit_non_parametric_gates_using_expectation_values(
        self, backend_for_gates_test, initial_gate, tested_gate, target_values
    ):

        if backend_for_gates_test.n_samples is None:
            pytest.xfail(
                "This test won't work for simulators without sampling, it should be covered by a test in QuantumSimulatorTests."
            )

        # Given
        qubit_list = [Qubit(0)]
        gate_1 = Gate(initial_gate, qubits=qubit_list)
        gate_2 = Gate(tested_gate, qubits=qubit_list)

        circuit = Circuit()
        circuit.qubits = qubit_list
        circuit.gates = [gate_1, gate_2]
        operators = [
            QubitOperator("[]"),
            QubitOperator("[X0]"),
            QubitOperator("[Y0]"),
            QubitOperator("[Z0]"),
        ]

        sigma = 1 / np.sqrt(backend_for_gates_test.n_samples)

        for i, operator in enumerate(operators):
            # When
            estimator = BasicEstimator()
            expectation_value = estimator.get_estimated_expectation_values(
                backend_for_gates_test,
                circuit,
                operator,
            ).values[0]

            # Then
            assert expectation_value == pytest.approx(target_values[i], abs=sigma * 3)

    @pytest.mark.parametrize(
        "initial_gate,tested_gate,params,target_values",
        one_qubit_parametric_gates_exp_vals_test_set,
    )
    def test_one_qubit_parametric_gates_using_expectation_values(
        self, backend_for_gates_test, initial_gate, tested_gate, params, target_values
    ):

        if backend_for_gates_test.n_samples is None:
            pytest.xfail(
                "This test won't work for simulators without sampling, it's covered by a test in QuantumSimulatorTests."
            )

        # Given
        qubit_list = [Qubit(0)]
        gate_1 = Gate(initial_gate, qubits=qubit_list)
        gate_2 = Gate(tested_gate, params=params, qubits=qubit_list)

        circuit = Circuit()
        circuit.qubits = qubit_list
        circuit.gates = [gate_1, gate_2]
        operators = [
            QubitOperator("[]"),
            QubitOperator("[X0]"),
            QubitOperator("[Y0]"),
            QubitOperator("[Z0]"),
        ]

        sigma = 1 / np.sqrt(backend_for_gates_test.n_samples)

        for i, operator in enumerate(operators):
            # When
            estimator = BasicEstimator()
            expectation_value = estimator.get_estimated_expectation_values(
                backend_for_gates_test,
                circuit,
                operator,
            ).values[0]

            # Then
            assert expectation_value == pytest.approx(target_values[i], abs=sigma * 3)

    @pytest.mark.parametrize(
        "initial_gates,tested_gate,operators,target_values",
        two_qubit_non_parametric_gates_exp_vals_test_set,
    )
    def test_two_qubit_non_parametric_gates_using_expectation_values(
        self,
        backend_for_gates_test,
        initial_gates,
        tested_gate,
        operators,
        target_values,
    ):

        if backend_for_gates_test.n_samples is None:
            pytest.xfail(
                "This test won't work for simulators without sampling, it's covered by a test in QuantumSimulatorTests."
            )

        # Given
        qubit_list = [Qubit(0), Qubit(1)]
        gate_1 = Gate(initial_gates[0], qubits=[qubit_list[0]])
        gate_2 = Gate(initial_gates[1], qubits=[qubit_list[1]])
        gate_3 = Gate(tested_gate, qubits=qubit_list)

        circuit = Circuit()
        circuit.qubits = qubit_list
        circuit.gates = [gate_1, gate_2, gate_3]

        sigma = 1 / np.sqrt(backend_for_gates_test.n_samples)

        for i, operator in enumerate(operators):
            # When
            operator = QubitOperator(operator)
            estimator = BasicEstimator()
            expectation_value = estimator.get_estimated_expectation_values(
                backend_for_gates_test,
                circuit,
                operator,
            ).values[0]

            # Then
            assert expectation_value == pytest.approx(target_values[i], abs=sigma * 5)

    @pytest.mark.parametrize(
        "initial_gates,tested_gate,params,operators,target_values",
        two_qubit_parametric_gates_exp_vals_test_set,
    )
    def test_two_qubit_parametric_gates_using_expectation_values(
        self,
        backend_for_gates_test,
        initial_gates,
        tested_gate,
        params,
        operators,
        target_values,
    ):

        if backend_for_gates_test.n_samples is None:
            pytest.xfail(
                "This test won't work for simulators without sampling, it's covered by a test in QuantumSimulatorTests."
            )

        # Given
        qubit_list = [Qubit(0), Qubit(1)]
        gate_1 = Gate(initial_gates[0], qubits=[qubit_list[0]])
        gate_2 = Gate(initial_gates[1], qubits=[qubit_list[1]])
        gate_3 = Gate(tested_gate, params=params, qubits=qubit_list)

        circuit = Circuit()
        circuit.qubits = qubit_list
        circuit.gates = [gate_1, gate_2, gate_3]

        sigma = 1 / np.sqrt(backend_for_gates_test.n_samples)

        for i, operator in enumerate(operators):
            # When
            operator = QubitOperator(operator)
            estimator = BasicEstimator()
            expectation_value = estimator.get_estimated_expectation_values(
                backend_for_gates_test,
                circuit,
                operator,
            ).values[0]

            # Then
            assert expectation_value == pytest.approx(target_values[i], abs=sigma * 5)


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


class QuantumSimulatorGatesTest:
    @pytest.mark.parametrize(
        "initial_gate,tested_gate,target_amplitudes",
        one_qubit_non_parametric_gates_amplitudes_test_set,
    )
    def test_one_qubit_non_parametric_gates_using_amplitudes(
        self, wf_simulator, initial_gate, tested_gate, target_amplitudes
    ):
        # Given
        qubit_list = [Qubit(0)]
        gate_1 = Gate(initial_gate, qubits=qubit_list)
        gate_2 = Gate(tested_gate, qubits=qubit_list)

        circuit = Circuit()
        circuit.qubits = qubit_list
        circuit.gates = [gate_1, gate_2]

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gate,tested_gate,params,target_amplitudes",
        one_qubit_parametric_gates_amplitudes_test_set,
    )
    def test_one_qubit_parametric_gates_using_amplitudes(
        self, wf_simulator, initial_gate, tested_gate, params, target_amplitudes
    ):
        # Given
        qubit_list = [Qubit(0)]
        gate_1 = Gate(initial_gate, qubits=qubit_list)
        gate_2 = Gate(tested_gate, params=params, qubits=qubit_list)

        circuit = Circuit()
        circuit.qubits = qubit_list
        circuit.gates = [gate_1, gate_2]

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gates,tested_gate,target_amplitudes",
        two_qubit_non_parametric_gates_amplitudes_test_set,
    )
    def test_two_qubit_non_parametric_gates_using_amplitudes(
        self, wf_simulator, initial_gates, tested_gate, target_amplitudes
    ):
        # Given
        qubit_list = [Qubit(0), Qubit(1)]
        gate_1 = Gate(initial_gates[0], qubits=[qubit_list[0]])
        gate_2 = Gate(initial_gates[1], qubits=[qubit_list[1]])
        gate_3 = Gate(tested_gate, qubits=qubit_list)

        circuit = Circuit()
        circuit.qubits = qubit_list
        circuit.gates = [gate_1, gate_2, gate_3]

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gates,tested_gate,params,target_amplitudes",
        two_qubit_parametric_gates_amplitudes_test_set,
    )
    def test_two_qubit_parametric_gates_using_amplitudes(
        self, wf_simulator, initial_gates, tested_gate, params, target_amplitudes
    ):
        # Given
        qubit_list = [Qubit(0), Qubit(1)]
        gate_1 = Gate(initial_gates[0], qubits=[qubit_list[0]])
        gate_2 = Gate(initial_gates[1], qubits=[qubit_list[1]])
        gate_3 = Gate(tested_gate, params=params, qubits=qubit_list)

        circuit = Circuit()
        circuit.qubits = qubit_list

        circuit.gates = [gate_1, gate_2, gate_3]
        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)
