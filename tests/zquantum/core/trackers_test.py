import json
from functools import partial
from os import remove

import networkx as nx
import numpy as np
import pytest
from qeqiskit.simulator import QiskitSimulator

# from qequlacs.simulator import QulacsSimulator
from zquantum.core.cost_function import AnsatzBasedCostFunction
from zquantum.core.estimation import (
    allocate_shots_uniformly,
    estimate_expectation_values_by_averaging,
)
from zquantum.core.trackers import MeasurementTrackingBackend
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.problems.maxcut import MaxCut


@pytest.fixture
def H():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(0, 1, weight=10)
    G.add_edge(0, 3, weight=10)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    problem = MaxCut()
    return problem.get_hamiltonian(G)


class TestMeasurementTrackingBackend:
    def test_save_measurement_data_from_maxcut_qaoa(self, H):
        # Given
        ansatz = QAOAFarhiAnsatz(1, cost_hamiltonian=H)
        backend = MeasurementTrackingBackend(
            inner_backend=QiskitSimulator(device_name="aer_simulator")
        )

        estimation_method = estimate_expectation_values_by_averaging
        shot_allocation = partial(allocate_shots_uniformly, number_of_shots=100)
        estimation_preprocessors = [shot_allocation]
        optimizer = ScipyOptimizer(method="L-BFGS-B", options={"maxiter": 10})
        cost_function = AnsatzBasedCostFunction(
            H, ansatz, backend, estimation_method, estimation_preprocessors
        )
        initial_params = np.array([0, 0])
        # When
        opt_results = optimizer.minimize(cost_function, initial_params)
        circuit = ansatz.get_executable_circuit(opt_results.opt_params)
        backend.n_samples = 10000
        backend.run_circuit_and_measure(circuit, n_samples=1000)
        f = open(backend.file_name)
        data = json.load(f)
        # Then
        assert data["raw-measurement-data"][0]["device"] == "QiskitSimulator"
        assert data["raw-measurement-data"][0]["circuit"]["n_qubits"] == 4
        assert data["raw-measurement-data"][0]["number_of_shots"] == 1000
        """Assert solutions are in the recorded data"""
        assert "1000" in data["raw-measurement-data"][0]["counts"]
        assert "0101" in data["raw-measurement-data"][0]["counts"]
        # Cleanup
        remove(backend.file_name)

    def test_save_measurement_data_while_inner_backend_used_elsewhere(self, H):
        # Given
        ansatz = QAOAFarhiAnsatz(1, cost_hamiltonian=H)
        inner = QiskitSimulator(device_name="aer_simulator")
        tracker_backend = MeasurementTrackingBackend(inner_backend=inner)

        estimation_method = estimate_expectation_values_by_averaging
        shot_allocation = partial(allocate_shots_uniformly, number_of_shots=100)
        estimation_preprocessors = [shot_allocation]
        optimizer = ScipyOptimizer(method="L-BFGS-B", options={"maxiter": 10})
        cost_function = AnsatzBasedCostFunction(
            H, ansatz, tracker_backend, estimation_method, estimation_preprocessors
        )
        cost_function = AnsatzBasedCostFunction(
            H, ansatz, inner, estimation_method, estimation_preprocessors
        )
        initial_params = np.array([0, 0])
        # When
        opt_results = optimizer.minimize(cost_function, initial_params)
        circuit = ansatz.get_executable_circuit(opt_results.opt_params)
        tracker_backend.n_samples = 10000
        inner.n_samples = 10000
        tracker_backend.run_circuit_and_measure(circuit, n_samples=1000)
        inner.run_circuit_and_measure(circuit, n_samples=1000)
        f = open(tracker_backend.file_name)
        data = json.load(f)
        # Then
        assert data["raw-measurement-data"][0]["device"] == "QiskitSimulator"
        assert data["raw-measurement-data"][0]["circuit"]["n_qubits"] == 4
        assert data["raw-measurement-data"][0]["number_of_shots"] == 1000
        """Assert solutions are in the recorded data"""
        assert "1000" in data["raw-measurement-data"][0]["counts"]
        assert "0101" in data["raw-measurement-data"][0]["counts"]
        # Cleanup
        remove(tracker_backend.file_name)

    def test_save_measurement_data_when_inner_backend_is_reused(self, H):
        # Given
        ansatz = QAOAFarhiAnsatz(1, cost_hamiltonian=H)
        inner = QiskitSimulator(device_name="aer_simulator")
        backend_1 = MeasurementTrackingBackend(inner_backend=inner)
        backend_2 = MeasurementTrackingBackend(inner_backend=inner)

        estimation_method = estimate_expectation_values_by_averaging
        shot_allocation = partial(allocate_shots_uniformly, number_of_shots=100)
        estimation_preprocessors = [shot_allocation]
        optimizer = ScipyOptimizer(method="L-BFGS-B", options={"maxiter": 10})
        cost_function_1 = AnsatzBasedCostFunction(
            H, ansatz, backend_1, estimation_method, estimation_preprocessors
        )
        cost_function_2 = AnsatzBasedCostFunction(
            H, ansatz, backend_2, estimation_method, estimation_preprocessors
        )
        initial_params = np.array([0, 0])
        # When
        opt_results_1 = optimizer.minimize(cost_function_1, initial_params)
        opt_results_2 = optimizer.minimize(cost_function_2, initial_params)
        circuit_1 = ansatz.get_executable_circuit(opt_results_1.opt_params)
        circuit_2 = ansatz.get_executable_circuit(opt_results_2.opt_params)
        backend_1.n_samples = 10000
        backend_2.n_samples = 10000
        backend_1.run_circuit_and_measure(circuit_1, n_samples=1000)
        backend_2.run_circuit_and_measure(circuit_2, n_samples=1000)
        f1 = open(backend_1.file_name)
        f2 = open(backend_2.file_name)
        data1 = json.load(f1)
        data2 = json.load(f2)
        # Then
        assert data1["raw-measurement-data"][0]["device"] == "QiskitSimulator"
        assert data1["raw-measurement-data"][0]["circuit"]["n_qubits"] == 4
        assert data1["raw-measurement-data"][0]["number_of_shots"] == 1000
        assert data2["raw-measurement-data"][0]["device"] == "QiskitSimulator"
        assert data2["raw-measurement-data"][0]["circuit"]["n_qubits"] == 4
        assert data2["raw-measurement-data"][0]["number_of_shots"] == 1000
        """Assert solutions are in the recorded data"""
        assert "1000" in data1["raw-measurement-data"][0]["counts"]
        assert "0101" in data1["raw-measurement-data"][0]["counts"]
        assert "1000" in data2["raw-measurement-data"][0]["counts"]
        assert "0101" in data2["raw-measurement-data"][0]["counts"]
        # Cleanup
        remove(backend_1.file_name)
        remove(backend_2.file_name)
