"""Test case prototypes that can be used in other projects.

Note that this file won't be executed on its own by pytest.
You need to define your own test cases that inherit from the ones defined here.

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
this process as a kind of "quantum process tomography" for gate unit testing.
Mathematically, correctness is ensured if the span of the input and outputs spans the
full vector space.  Checking a tomographically complete set of input and outputs could
be time consuming, especially in the case of sampling. Furthermore, we expect that the
bugs that will occur will lead to an effect on many inputs (rather than, say, a single
input-output pair).  Therefore, we are taking here a slightly lazy, but efficient
approach to testing these gates by testing how they transform a tomographically
incomplete set of input and outputs.

Gates tests use `backend_for_gates_test` instead of `backend` as an input parameter
because:
a) it has high chance of failing for noisy backends
b) having execution time in mind it's a good idea to use lower number of samples.
"""

import functools
from typing import List

import numpy as np
import pytest
from openfermion import QubitOperator
from zquantum.core.interfaces.backend import QuantumSimulator
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.wavefunction import Wavefunction

from ..bitstring_distribution import BitstringDistribution
from ..circuits import CNOT, Circuit, H, X, builtin_gate_by_name
from ..estimation import estimate_expectation_values_by_averaging
from ..measurement import ExpectationValues, Measurements
from ..testing.test_cases_for_backend_tests import (
    one_qubit_non_parametric_gates_amplitudes_test_set,
    one_qubit_non_parametric_gates_exp_vals_test_set,
    one_qubit_parametric_gates_amplitudes_test_set,
    one_qubit_parametric_gates_exp_vals_test_set,
    two_qubit_non_parametric_gates_amplitudes_test_set,
    two_qubit_non_parametric_gates_exp_vals_test_set,
    two_qubit_parametric_gates_amplitudes_test_set,
    two_qubit_parametric_gates_exp_vals_test_set,
)


def skip_tests_for_excluded_gates(func):
    @functools.wraps(func)
    def _wrapper(self, **kwargs):
        tested_gate = kwargs["tested_gate"]
        if tested_gate not in self.gates_to_exclude:
            func(self, **kwargs)
        else:
            pytest.xfail(f"{tested_gate} gate is excluded for tests for this backend.")

    return _wrapper


class QuantumBackendTests:
    def test_run_circuit_and_measure_correct_indexing(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        n_samples = 100
        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples)

        # Then (since SPAM error could result in unexpected bitstrings, we make sure the
        # most common bitstring is the one we expect)
        counts = measurements.get_counts()
        assert max(counts, key=counts.get) == "001"
        assert backend.number_of_circuits_run == 1
        assert backend.number_of_jobs_run == 1

    @pytest.mark.parametrize("n_samples", [-1, 0, 100.2, 1000.0])
    def test_run_circuit_and_measure_fails_for_invalid_n_samples(
        self, backend, n_samples
    ):
        # Given
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        with pytest.raises(AssertionError):
            backend.run_circuit_and_measure(circuit, n_samples)

    @pytest.mark.parametrize("n_samples", [1, 2, 10, 100])
    def test_run_circuit_and_measure_correct_num_measurements_attribute(
        self, backend, n_samples
    ):
        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples)

        # Then
        assert isinstance(measurements, Measurements)
        assert len(measurements.bitstrings) == n_samples
        assert backend.number_of_circuits_run == 1
        assert backend.number_of_jobs_run == 1

    @pytest.mark.parametrize("n_samples", [1, 2, 10, 100])
    def test_run_circuit_and_measure_correct_num_measurements_argument(
        self, backend, n_samples
    ):
        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples)

        # Then
        assert isinstance(measurements, Measurements)
        assert len(measurements.bitstrings) == n_samples
        assert backend.number_of_circuits_run == 1
        assert backend.number_of_jobs_run == 1

    def test_if_all_measurements_have_the_same_number_of_bits(self, backend):
        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples=100)

        # Then
        assert all(len(bitstring) == 3 for bitstring in measurements.bitstrings)
        assert backend.number_of_circuits_run == 1
        assert backend.number_of_jobs_run == 1

    def test_run_circuitset_and_measure(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        n_samples = 100
        number_of_circuits = 25
        # When
        n_samples_per_circuit = [n_samples] * number_of_circuits
        measurements_set = backend.run_circuitset_and_measure(
            [circuit] * number_of_circuits, n_samples_per_circuit
        )

        # Then (since SPAM error could result in unexpected bitstrings, we make sure the
        # most common bitstring is the one we expect)
        for measurements in measurements_set:
            counts = measurements.get_counts()
            assert max(counts, key=counts.get) == "001"
        assert backend.number_of_circuits_run == number_of_circuits

        if backend.supports_batching:
            assert backend.number_of_jobs_run == int(
                np.ceil(number_of_circuits / backend.batch_size)
            )
        else:
            assert backend.number_of_jobs_run == number_of_circuits

    def test_run_circuitset_and_measure_n_samples(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        first_circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        second_circuit = Circuit([X(0), X(1), X(2)])
        n_samples_per_circuit = [100, 105]

        # When
        measurements_set = backend.run_circuitset_and_measure(
            [first_circuit, second_circuit], n_samples_per_circuit
        )

        # Then (since SPAM error could result in unexpected bitstrings, we make sure the
        # most common bitstring is the one we expect)
        counts = measurements_set[0].get_counts()
        assert max(counts, key=counts.get) == "001"
        counts = measurements_set[1].get_counts()
        assert max(counts, key=counts.get) == "111"

        assert len(measurements_set[0].bitstrings) == n_samples_per_circuit[0]
        assert len(measurements_set[1].bitstrings) == n_samples_per_circuit[1]

        assert backend.number_of_circuits_run == 2

    def test_get_bitstring_distribution(self, backend):
        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])
        n_samples = 1000

        # When
        bitstring_distribution = backend.get_bitstring_distribution(
            circuit, n_samples=n_samples
        )

        # Then
        assert isinstance(bitstring_distribution, BitstringDistribution)
        assert bitstring_distribution.get_qubits_number() == 3
        assert bitstring_distribution.distribution_dict["000"] > 1 / 3
        assert bitstring_distribution.distribution_dict["111"] > 1 / 3
        assert backend.number_of_circuits_run == 1
        assert backend.number_of_jobs_run == 1


class QuantumBackendGatesTests:
    gates_to_exclude: List[str] = []

    @pytest.mark.parametrize(
        "initial_gate,tested_gate,target_values",
        one_qubit_non_parametric_gates_exp_vals_test_set,
    )
    @skip_tests_for_excluded_gates
    def test_one_qubit_non_parametric_gates_using_expectation_values(
        self, backend_for_gates_test, initial_gate, tested_gate, target_values
    ):
        n_samples = 1000

        # Given
        gate_1 = builtin_gate_by_name(initial_gate)(0)
        gate_2 = builtin_gate_by_name(tested_gate)(0)

        circuit = Circuit([gate_1, gate_2])
        operators = [
            QubitOperator("[]"),
            QubitOperator("[X0]"),
            QubitOperator("[Y0]"),
            QubitOperator("[Z0]"),
        ]

        sigma = 1 / np.sqrt(n_samples)

        for i, operator in enumerate(operators):
            # When
            estimation_tasks = [EstimationTask(operator, circuit, n_samples)]
            expectation_values = estimate_expectation_values_by_averaging(
                backend_for_gates_test, estimation_tasks
            )
            calculated_value = expectation_values.values[0]

            # Then
            assert calculated_value == pytest.approx(target_values[i], abs=sigma * 3)

    @pytest.mark.parametrize(
        "initial_gate,tested_gate,params,target_values",
        one_qubit_parametric_gates_exp_vals_test_set,
    )
    @skip_tests_for_excluded_gates
    def test_one_qubit_parametric_gates_using_expectation_values(
        self, backend_for_gates_test, initial_gate, tested_gate, params, target_values
    ):
        n_samples = 1000
        # Given
        gate_1 = builtin_gate_by_name(initial_gate)(0)
        gate_2 = builtin_gate_by_name(tested_gate)(*params)(0)

        circuit = Circuit([gate_1, gate_2])
        operators = [
            QubitOperator("[]"),
            QubitOperator("[X0]"),
            QubitOperator("[Y0]"),
            QubitOperator("[Z0]"),
        ]

        sigma = 1 / np.sqrt(n_samples)

        for i, operator in enumerate(operators):
            # When
            estimation_tasks = [EstimationTask(operator, circuit, n_samples)]
            expectation_values = estimate_expectation_values_by_averaging(
                backend_for_gates_test, estimation_tasks
            )
            calculated_value = expectation_values.values[0]

            # Then
            assert calculated_value == pytest.approx(target_values[i], abs=sigma * 3)

    @pytest.mark.parametrize(
        "initial_gates,tested_gate,operators,target_values",
        two_qubit_non_parametric_gates_exp_vals_test_set,
    )
    @skip_tests_for_excluded_gates
    def test_two_qubit_non_parametric_gates_using_expectation_values(
        self,
        backend_for_gates_test,
        initial_gates,
        tested_gate,
        operators,
        target_values,
    ):
        n_samples = 1000

        # Given
        gate_1 = builtin_gate_by_name(initial_gates[0])(0)
        gate_2 = builtin_gate_by_name(initial_gates[1])(1)
        gate_3 = builtin_gate_by_name(tested_gate)(0, 1)

        circuit = Circuit([gate_1, gate_2, gate_3])

        sigma = 1 / np.sqrt(n_samples)

        for i, operator in enumerate(operators):
            # When
            operator = QubitOperator(operator)
            estimation_tasks = [EstimationTask(operator, circuit, n_samples)]
            expectation_values = estimate_expectation_values_by_averaging(
                backend_for_gates_test, estimation_tasks
            )
            calculated_value = expectation_values.values[0]

            # Then
            assert calculated_value == pytest.approx(target_values[i], abs=sigma * 5)

    @pytest.mark.parametrize(
        "initial_gates,tested_gate,params,operators,target_values",
        two_qubit_parametric_gates_exp_vals_test_set,
    )
    @skip_tests_for_excluded_gates
    def test_two_qubit_parametric_gates_using_expectation_values(
        self,
        backend_for_gates_test,
        initial_gates,
        tested_gate,
        params,
        operators,
        target_values,
    ):
        n_samples = 1000

        # Given
        gate_1 = builtin_gate_by_name(initial_gates[0])(0)
        gate_2 = builtin_gate_by_name(initial_gates[1])(1)
        gate_3 = builtin_gate_by_name(tested_gate)(*params)(0, 1)

        circuit = Circuit([gate_1, gate_2, gate_3])

        sigma = 1 / np.sqrt(n_samples)

        for i, operator in enumerate(operators):
            # When
            operator = QubitOperator(operator)
            estimation_tasks = [EstimationTask(operator, circuit, n_samples)]
            expectation_values = estimate_expectation_values_by_averaging(
                backend_for_gates_test, estimation_tasks
            )
            calculated_value = expectation_values.values[0]

            # Then
            assert calculated_value == pytest.approx(target_values[i], abs=sigma * 5)


class QuantumSimulatorTests(QuantumBackendTests):
    def test_get_wavefunction(self, wf_simulator):
        # Given
        wf_simulator.number_of_circuits_run = 0
        wf_simulator.number_of_jobs_run = 0
        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert isinstance(wavefunction, Wavefunction)
        assert len(wavefunction.probabilities()) == 8
        assert wavefunction[0] == pytest.approx((1 / np.sqrt(2) + 0j), abs=1e-7)
        assert wavefunction[7] == pytest.approx((1 / np.sqrt(2) + 0j), abs=1e-7)
        assert wf_simulator.number_of_circuits_run == 1
        assert wf_simulator.number_of_jobs_run == 1

    def test_get_bitstring_distribution_wf_simulators(self, wf_simulator):
        # Given
        wf_simulator.number_of_circuits_run = 0
        wf_simulator.number_of_jobs_run = 0
        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])

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
        assert wf_simulator.number_of_circuits_run == 1
        assert wf_simulator.number_of_jobs_run == 1

    def test_get_exact_expectation_values(self, wf_simulator):
        # Given
        wf_simulator.number_of_circuits_run = 0
        wf_simulator.number_of_jobs_run = 0
        circuit = Circuit([H(0), X(1)])
        operator = QubitOperator("[Z0] + 2[Z1]")
        target_expectation_values = ExpectationValues(np.array([0.0, -2.0]))

        # When
        expectation_values = wf_simulator.get_exact_expectation_values(
            circuit, operator
        )

        assert np.allclose(expectation_values.values, target_expectation_values.values)
        assert wf_simulator.number_of_circuits_run == 1
        assert wf_simulator.number_of_jobs_run == 1


class QuantumSimulatorGatesTest:
    gates_to_exclude: List[str] = []

    @pytest.mark.parametrize(
        "initial_gate,tested_gate,target_amplitudes",
        one_qubit_non_parametric_gates_amplitudes_test_set,
    )
    @skip_tests_for_excluded_gates
    def test_one_qubit_non_parametric_gates_using_amplitudes(
        self, wf_simulator, initial_gate, tested_gate, target_amplitudes
    ):
        # Given
        gate_1 = builtin_gate_by_name(initial_gate)(0)
        gate_2 = builtin_gate_by_name(tested_gate)(0)

        circuit = Circuit([gate_1, gate_2])

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gate,tested_gate,params,target_amplitudes",
        one_qubit_parametric_gates_amplitudes_test_set,
    )
    @skip_tests_for_excluded_gates
    def test_one_qubit_parametric_gates_using_amplitudes(
        self, wf_simulator, initial_gate, tested_gate, params, target_amplitudes
    ):
        # Given
        gate_1 = builtin_gate_by_name(initial_gate)(0)
        gate_2 = builtin_gate_by_name(tested_gate)(*params)(0)

        circuit = Circuit([gate_1, gate_2])

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gates,tested_gate,target_amplitudes",
        two_qubit_non_parametric_gates_amplitudes_test_set,
    )
    @skip_tests_for_excluded_gates
    def test_two_qubit_non_parametric_gates_using_amplitudes(
        self, wf_simulator, initial_gates, tested_gate, target_amplitudes
    ):
        # Given
        gate_1 = builtin_gate_by_name(initial_gates[0])(0)
        gate_2 = builtin_gate_by_name(initial_gates[1])(1)
        gate_3 = builtin_gate_by_name(tested_gate)(0, 1)

        circuit = Circuit([gate_1, gate_2, gate_3])

        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)

    @pytest.mark.parametrize(
        "initial_gates,tested_gate,params,target_amplitudes",
        two_qubit_parametric_gates_amplitudes_test_set,
    )
    @skip_tests_for_excluded_gates
    def test_two_qubit_parametric_gates_using_amplitudes(
        self, wf_simulator, initial_gates, tested_gate, params, target_amplitudes
    ):
        # Given
        gate_1 = builtin_gate_by_name(initial_gates[0])(0)
        gate_2 = builtin_gate_by_name(initial_gates[1])(1)
        gate_3 = builtin_gate_by_name(tested_gate)(*params)(0, 1)

        circuit = Circuit([gate_1, gate_2, gate_3])
        # When
        wavefunction = wf_simulator.get_wavefunction(circuit)

        # Then
        assert np.allclose(wavefunction.amplitudes, target_amplitudes)
