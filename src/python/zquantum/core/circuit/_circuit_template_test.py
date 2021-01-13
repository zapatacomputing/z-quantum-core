import unittest
import os
import numpy as np
import cirq
from ._circuit_template import (
    save_circuit_template,
    load_circuit_template,
    save_circuit_template_params,
    load_circuit_template_params,
    combine_ansatz_params,
    build_uniform_param_grid,
    save_parameter_grid,
    load_parameter_grid,
    CircuitLayers,
    save_circuit_layers,
    load_circuit_layers,
    ParameterGrid,
    save_circuit_ordering,
    load_circuit_ordering,
    save_circuit_connectivity,
    load_circuit_connectivity,
    build_circuit_layers_and_connectivity,
    CircuitConnectivity,
    create_layer_of_gates,
)
from . import Gate, Qubit, Circuit
from ..utils import SCHEMA_VERSION, compare_unitary
from scipy.optimize import OptimizeResult


class TestCircuitTemplate(unittest.TestCase):
    def test_circuit_template_io(self):
        # Given
        filename = "circuit_template.json"
        template = {
            "ansatz_type": "singlet UCCSD",
            "ansatz_module": "zquantum.vqe.ansatzes.ucc",
            "ansatz_func": "build_singlet_uccsd_circuit",
            "ansatz_kwargs": {
                "n_mo": 2,
                "n_electrons": 2,
                "transformation": "Jordan-Wigner",
            },
            "n_params": [2],
        }

        # When
        save_circuit_template(template, filename)
        recreated_template = load_circuit_template(filename)
        schema = recreated_template.pop("schema")

        # Then
        self.assertEqual(schema, SCHEMA_VERSION + "-circuit_template")
        self.assertDictEqual(template, recreated_template)

        os.remove(filename)

    def test_circuit_template_params_io(self):
        # Given
        params = np.random.rand((10))
        filename = "circuit_template_params.json"

        # When
        save_circuit_template_params(params, filename)
        recreated_params = load_circuit_template_params(filename)

        # Then
        np.testing.assert_array_equal(params, recreated_params)
        os.remove(filename)

    def test_combine_ansatz_params(self):
        # Given
        params1 = np.array([1.0, 2.0])
        params2 = np.array([3.0, 4.0])
        target_params = np.array([1.0, 2.0, 3.0, 4.0])

        # When
        combined_params = combine_ansatz_params(params1, params2)

        # Then
        self.assertTrue(np.allclose(combined_params, target_params))


class TestParameterGrid(unittest.TestCase):
    def test_dict_io(self):
        # Given
        param_ranges = [(0, 1, 0.1)] * 2
        grid = ParameterGrid(param_ranges)

        # When
        grid_dict = grid.to_dict()
        recreated_grid = ParameterGrid.from_dict(grid_dict)

        # Then
        np.testing.assert_array_equal(recreated_grid.param_ranges, grid.param_ranges)

    def test_params_list(self):
        # Given
        param_ranges = [(0, 1, 0.5)] * 2
        grid = ParameterGrid(param_ranges)
        correct_params_list = [
            np.array([0, 0]),
            np.array([0, 0.5]),
            np.array([0.5, 0]),
            np.array([0.5, 0.5]),
        ]

        # When
        params_list = grid.params_list

        # Then
        np.testing.assert_array_equal(params_list, correct_params_list)

    def test_params_meshgrid(self):
        # Given
        param_ranges = [(0, 1, 0.5)] * 2
        grid = ParameterGrid(param_ranges)
        correct_params_meshgrid = [
            np.array([[0, 0], [0.5, 0.5]]),
            np.array([[0, 0.5], [0, 0.5]]),
        ]

        # When
        params_meshgrid = grid.params_meshgrid

        # Then
        np.testing.assert_array_equal(params_meshgrid, correct_params_meshgrid)

    def test_parameter_grid_io(self):
        # Given
        param_ranges = [(0, 1, 0.1)] * 2
        grid = ParameterGrid(param_ranges)

        # When
        save_parameter_grid(grid, "grid.json")
        loaded_grid = load_parameter_grid("grid.json")

        # Then
        self.assertEqual(len(grid.param_ranges), len(loaded_grid.param_ranges))
        for i in range(len(grid.param_ranges)):
            self.assertEqual(
                tuple(grid.param_ranges[i]), tuple(loaded_grid.param_ranges[i])
            )
        os.remove("grid.json")

    def test_build_uniform_param_grid(self):
        # Given
        n_params_per_layer = 2

        # When
        grid = build_uniform_param_grid(
            n_params_per_layer,
            n_layers=1,
            min_value=0.0,
            max_value=2 * np.pi,
            step=np.pi / 5,
        )

        # Then
        for i in range(10):
            for j in range(10):
                self.assertAlmostEqual(grid.params_list[i + 10 * j][0], j * np.pi / 5)
                self.assertAlmostEqual(grid.params_list[i + 10 * j][1], i * np.pi / 5)
                print(f"{i} {j} {grid.params_meshgrid[0][i, j]}")
                self.assertAlmostEqual(grid.params_meshgrid[0][i, j], i * np.pi / 5)
                self.assertAlmostEqual(grid.params_meshgrid[1][i, j], j * np.pi / 5)


class TestCircuitLayers(unittest.TestCase):
    def test_dict_io(self):
        # Given
        layers = CircuitLayers([[(0, 1), (2, 3)], [(1, 2), (3, 0)]])

        # When
        layers_dict = layers.to_dict()
        recreated_layers = CircuitLayers.from_dict(layers_dict)

        # Then
        self.assertEqual(len(layers.layers), len(recreated_layers.layers))
        for layer, recreated_layer in zip(layers.layers, recreated_layers.layers):
            self.assertEqual(layer, recreated_layer)

    def test_circuit_layers_io(self):
        # Given
        layers = CircuitLayers([[(0, 1), (2, 3)], [(1, 2), (3, 0)]])

        # When
        save_circuit_layers(layers, "layers.json")
        loaded_layers = load_circuit_layers("layers.json")

        # Then
        self.assertEqual(len(layers.layers), len(loaded_layers.layers))
        for layer, loaded_layer in zip(layers.layers, loaded_layers.layers):
            self.assertEqual(layer, loaded_layer)
        os.remove("layers.json")

    def test_circuit_ordering_io(self):
        # Given
        ordering = [0, 1, 3, 2]

        # When
        save_circuit_ordering(ordering, "ordering.json")
        loaded_ordering = load_circuit_ordering("ordering.json")

        # Then
        self.assertEqual(loaded_ordering, ordering)
        os.remove("ordering.json")

    def test_circuit_connectivity_io(self):
        # Given
        connectivity = CircuitConnectivity([(0, 1), (1, 2), (2, 3), (3, 0)])
        # When
        save_circuit_connectivity(connectivity, "connectivity.json")
        loaded_connectivity = load_circuit_connectivity("connectivity.json")
        # Then
        self.assertEqual(
            len(connectivity.connectivity), len(loaded_connectivity.connectivity)
        )
        for connection, loaded_connection in zip(
            connectivity.connectivity, loaded_connectivity.connectivity
        ):
            self.assertEqual(connection, loaded_connection)
        os.remove("connectivity.json")

    def test_build_circuit_layers_and_connectivity(self):
        # Sycamore
        # Given
        test_layers = [
            [(3, 1), (4, 7)],
            [(3, 6), (4, 2), (5, 8)],
            [(3, 1), (4, 2)],
            [(3, 6), (4, 7), (5, 8)],
            [(3, 0), (4, 8), (5, 2)],
            [(3, 7), (4, 1)],
            [(3, 0), (4, 1), (5, 2)],
            [(3, 7), (4, 8)],
        ]
        test_connectivity = [
            (3, 0),
            (3, 1),
            (4, 1),
            (4, 2),
            (5, 2),
            (3, 6),
            (3, 7),
            (4, 7),
            (4, 8),
            (5, 8),
        ]
        # When
        connectivity, layers = build_circuit_layers_and_connectivity(3, 3, "sycamore")

        # Then
        for layer, test_layer in zip(layers.layers, test_layers):
            self.assertEqual(layer, test_layer)

        for row, test_row in zip(connectivity.connectivity, test_connectivity):
            self.assertEqual(row, test_row)

        # Nearest-Neighbors
        # Given
        test_layers = [[(0, 1), (2, 3)], [(1, 2)]]
        test_connectivity = [(0, 1), (2, 3), (1, 2)]

        # When
        connectivity, layers = build_circuit_layers_and_connectivity(
            4, "nearest-neighbor"
        )

        # Then
        for layer, test_layer in zip(layers.layers, test_layers):
            self.assertEqual(layer, test_layer)

        for row, test_row in zip(connectivity.connectivity, test_connectivity):
            self.assertEqual(row, test_row)

    def test_create_layer_of_gates_not_parametrized(self):
        # Given
        number_of_qubits = 4
        gate_name = "X"
        qubits = [Qubit(i) for i in range(0, number_of_qubits)]
        gate_0 = Gate(gate_name, qubits=[qubits[0]])
        gate_1 = Gate(gate_name, qubits=[qubits[1]])
        gate_2 = Gate(gate_name, qubits=[qubits[2]])
        gate_3 = Gate(gate_name, qubits=[qubits[3]])
        target_circuit = Circuit()
        target_circuit.qubits = qubits
        target_circuit.gates = [gate_0, gate_1, gate_2, gate_3]

        # When
        layer_of_x = create_layer_of_gates(number_of_qubits, gate_name)

        # Then
        self.assertEqual(layer_of_x, target_circuit)

    def test_create_layer_of_gates_parametrized(self):
        # Given
        single_qubit_gate = "Ry"
        n_qubits_list = [2, 3, 4, 10]

        for n_qubits in n_qubits_list:
            # Given
            params = [x for x in range(0, n_qubits)]
            test = cirq.Circuit()
            qubits = [cirq.LineQubit(x) for x in range(0, n_qubits)]
            for i in range(0, n_qubits):
                test.append(cirq.ry(params[i]).on(qubits[i]))
            u_cirq = test._unitary_()

            # When
            circ = create_layer_of_gates(n_qubits, single_qubit_gate, params)
            unitary = circ.to_cirq()._unitary_()

            # Then
            self.assertEqual(circ.n_multiqubit_gates, 0)
            self.assertEqual(compare_unitary(unitary, u_cirq, tol=1e-10), True)

    def test_create_layer_of_gates_wrong_num_params(self):
        # Given
        single_qubit_gate = "Ry"
        n_qubits = 2
        params = np.ones(3)
        # When/Then
        self.assertRaises(
            AssertionError,
            lambda: create_layer_of_gates(n_qubits, single_qubit_gate, params),
        )
