import copy
import json
import os
import sys

import numpy as np
import pytest
from zquantum.core.circuit import Circuit
from zquantum.core.circuit import (
    add_ancilla_register_to_circuit as _add_ancilla_register_to_circuit,
)
from zquantum.core.circuit import (
    build_circuit_layers_and_connectivity as _build_circuit_layers_and_connectivity,
)
from zquantum.core.circuit import build_uniform_param_grid as _build_uniform_param_grid
from zquantum.core.circuit import (
    load_circuit,
    load_circuit_connectivity,
    load_circuit_layers,
    load_circuit_set,
    load_circuit_template_params,
    load_parameter_grid,
    save_circuit,
    save_circuit_set,
    save_circuit_template_params,
)
from zquantum.core.testing import create_random_circuit as _create_random_circuit
from zquantum.core.utils import RNDSEED, create_object

sys.path.append("../..")
from steps.circuit import (
    add_ancilla_register_to_circuit,
    batch_circuits,
    build_ansatz_circuit,
    build_circuit_layers_and_connectivity,
    build_uniform_param_grid,
    combine_ansatz_params,
    concatenate_circuits,
    create_random_circuit,
    generate_random_ansatz_params,
)


def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


class TestGenerateRandomAnsatzParams:
    @pytest.mark.parametrize(
        "number_of_layers",
        [0, 1, 4, 7],
    )
    def test_generate_random_ansatz_params_using_mock_ansatz_specs(
        self, number_of_layers
    ):
        # Given
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": number_of_layers,
            "problem_size": 2,
        }
        seed = RNDSEED

        filename = "params.json"
        remove_file_if_exists(filename)

        # When
        generate_random_ansatz_params(ansatz_specs=ansatz_specs, seed=seed)

        # Then
        try:
            parameters = load_circuit_template_params(filename)
            assert len(parameters) == number_of_layers
        finally:
            remove_file_if_exists(filename)

    def test_generate_random_ansatz_params_ansatz_specs_as_string(self):
        # Given
        number_of_layers = 5
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": number_of_layers,
            "problem_size": 2,
        }
        seed = RNDSEED

        filename = "params.json"
        remove_file_if_exists(filename)

        # When
        generate_random_ansatz_params(ansatz_specs=json.dumps(ansatz_specs), seed=seed)

        # Then
        try:
            parameters = load_circuit_template_params(filename)
            assert len(parameters) == number_of_layers
        finally:
            remove_file_if_exists(filename)

    @pytest.mark.parametrize(
        "number_of_parameters",
        [i for i in range(12)],
    )
    def test_generate_random_ansatz_params_using_number_of_parameters(
        self,
        number_of_parameters,
    ):
        # Given
        seed = RNDSEED

        filename = "params.json"
        remove_file_if_exists(filename)

        # When
        generate_random_ansatz_params(
            number_of_parameters=number_of_parameters, seed=seed
        )

        # Then
        try:
            parameters = load_circuit_template_params(filename)
            assert len(parameters) == number_of_parameters
        finally:
            remove_file_if_exists(filename)

    def test_generate_random_ansatz_params_fails_with_both_ansatz_specs_and_number_of_parameters(
        self,
    ):
        number_of_parameters = 2
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": 2,
            "problem_size": 1,
        }
        seed = RNDSEED
        filename = "params.json"

        try:
            with pytest.raises(AssertionError):
                generate_random_ansatz_params(
                    ansatz_specs=ansatz_specs,
                    number_of_parameters=number_of_parameters,
                    seed=seed,
                )
        finally:
            remove_file_if_exists(filename)

    def test_generate_random_ansatz_params_fails_with_neither_ansatz_specs_nor_number_of_parameters(
        self,
    ):
        seed = RNDSEED
        filename = "params.json"

        try:
            with pytest.raises(AssertionError):
                generate_random_ansatz_params(
                    seed=seed,
                )
        finally:
            remove_file_if_exists(filename)


class TestCombineAnsatzParams:
    @pytest.fixture(
        params=[
            ([], []),
            ([1.0], []),
            ([], [1.0]),
            ([0.0], [1.0]),
            ([0.0, 1.0, 3.0, 5.0, -2.3], [1.0]),
        ]
    )
    def params_filenames(self, request):
        params1_filename = "params1.json"
        save_circuit_template_params(np.array(request.param[0]), params1_filename)

        params2_filename = "params2.json"
        save_circuit_template_params(np.array(request.param[1]), params2_filename)

        yield (params1_filename, params2_filename)

        remove_file_if_exists(params1_filename)
        remove_file_if_exists(params2_filename)

    def test_combine_ansatz_params(self, params_filenames):
        # Given
        params1_filename, params2_filename = params_filenames

        # When
        combine_ansatz_params(params1_filename, params2_filename)

        # Then
        try:
            combined_parameters_filename = "combined-params.json"
            parameters = load_circuit_template_params(combined_parameters_filename)
            params1 = load_circuit_template_params(params1_filename)
            params2 = load_circuit_template_params(params2_filename)
            assert all(parameters == np.concatenate([params1, params2]))
        finally:
            remove_file_if_exists(combined_parameters_filename)


class TestBuildAnsatzCircuit:
    @pytest.fixture(params=[0, 1, 2, 5])
    def number_of_layers(self, request):
        return request.param

    @pytest.fixture()
    def params_filename_and_number_of_layers(self, number_of_layers):
        params = np.random.uniform(low=0, high=np.pi, size=number_of_layers)
        params_filename = "params.json"
        save_circuit_template_params(np.array(params), params_filename)

        yield params_filename, number_of_layers

        remove_file_if_exists(params_filename)

    def test_build_ansatz_circuit_with_parameter_values(
        self, params_filename_and_number_of_layers
    ):
        # Given
        params_filename, number_of_layers = params_filename_and_number_of_layers

        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": number_of_layers,
            "problem_size": 2,
        }

        parameters = load_circuit_template_params(params_filename)
        ansatz = create_object(copy.deepcopy(ansatz_specs))
        expected_circuit = ansatz.get_executable_circuit(parameters)

        # When
        build_ansatz_circuit(ansatz_specs=ansatz_specs, params=params_filename)

        # Then
        try:
            circuit_filename = "circuit.json"
            circuit = load_circuit(circuit_filename)
            assert isinstance(circuit, Circuit)
            assert circuit == expected_circuit
        finally:
            remove_file_if_exists(circuit_filename)

    def test_build_ansatz_circuit_without_parameter_values(self, number_of_layers):
        # Given
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": number_of_layers,
            "problem_size": 2,
        }

        ansatz = create_object(copy.deepcopy(ansatz_specs))
        expected_circuit = ansatz.parametrized_circuit

        # When
        build_ansatz_circuit(ansatz_specs=ansatz_specs)

        # Then
        try:
            circuit_filename = "circuit.json"
            circuit = load_circuit(circuit_filename)
            assert isinstance(circuit, Circuit)
            assert circuit == expected_circuit
        finally:
            remove_file_if_exists(circuit_filename)

    def test_build_ansatz_circuit_ansatz_specs_as_string(self):
        # Given
        number_of_layers = 2
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": number_of_layers,
            "problem_size": 2,
        }

        ansatz = create_object(copy.deepcopy(ansatz_specs))
        expected_circuit = ansatz.parametrized_circuit

        # When
        build_ansatz_circuit(ansatz_specs=json.dumps(ansatz_specs))

        # Then
        try:
            circuit_filename = "circuit.json"
            circuit = load_circuit(circuit_filename)
            assert isinstance(circuit, Circuit)
            assert circuit == expected_circuit
        finally:
            remove_file_if_exists(circuit_filename)

    def test_build_ansatz_circuit_raises_exception_on_invalid_inputs(self):
        params_filename = "params.json"
        save_circuit_template_params(np.array([1.0]), params_filename)

        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": 0,
            "problem_size": 2,
        }

        try:
            circuit_filename = "circuit.json"
            with pytest.raises(Exception):
                build_ansatz_circuit(ansatz_specs=ansatz_specs, params=params_filename)
        finally:
            remove_file_if_exists(params_filename)
            remove_file_if_exists(circuit_filename)


class TestBuildUniformParameterGrid:
    @pytest.mark.parametrize(
        "number_of_ansatz_layers, problem_size, number_of_layers, min_value, max_value, step",
        [
            (0, 2, 2, 0, 1, 0.5),
            (1, 2, 2, 0, 1, 0.5),
            (1, 0, 2, 0, 1, 0.5),
            (6, 2, 2, 0, 1, 0.5),
            (1, 2, 6, 0, 1, 0.5),
            (1, 2, 6, -np.pi, 1, 0.5),
            (1, 2, 6, -np.pi, np.pi, 0.5),
            (1, 2, 6, 0, 1, 0.01),
        ],
    )
    def test_build_uniform_param_grid_ansatz_specs(
        self,
        number_of_ansatz_layers,
        problem_size,
        number_of_layers,
        min_value,
        max_value,
        step,
    ):
        # Given
        expected_parameter_grid_filename = "parameter-grid.json"
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": number_of_ansatz_layers,
            "problem_size": problem_size,
        }
        ansatz = create_object(copy.deepcopy(ansatz_specs))
        expected_parameter_grid = _build_uniform_param_grid(
            ansatz.number_of_params,
            number_of_layers,
            min_value=min_value,
            max_value=max_value,
            step=step,
        )

        # When
        build_uniform_param_grid(
            ansatz_specs=ansatz_specs,
            number_of_layers=number_of_layers,
            min_value=min_value,
            max_value=max_value,
            step=step,
        )

        # Then
        try:
            parameter_grid = load_parameter_grid(expected_parameter_grid_filename)
            assert [
                tuple(param) for param in parameter_grid.param_ranges
            ] == expected_parameter_grid.param_ranges
        finally:
            remove_file_if_exists(expected_parameter_grid_filename)

    @pytest.mark.parametrize(
        "number_of_ansatz_layers, problem_size, number_of_layers, min_value, max_value, step",
        [
            (0, 2, 2, 0, 1, 0.5),
            (1, 2, 2, 0, 1, 0.5),
            (1, 0, 2, 0, 1, 0.5),
            (6, 2, 2, 0, 1, 0.5),
            (1, 2, 6, 0, 1, 0.5),
            (1, 2, 6, -np.pi, 1, 0.5),
            (1, 2, 6, -np.pi, np.pi, 0.5),
            (1, 2, 6, 0, 1, 0.01),
        ],
    )
    def test_build_uniform_param_grid_ansatz_specs_as_string(
        self,
        number_of_ansatz_layers,
        problem_size,
        number_of_layers,
        min_value,
        max_value,
        step,
    ):
        # Given
        expected_parameter_grid_filename = "parameter-grid.json"
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": number_of_ansatz_layers,
            "problem_size": problem_size,
        }
        ansatz = create_object(copy.deepcopy(ansatz_specs))
        expected_parameter_grid = _build_uniform_param_grid(
            ansatz.number_of_params,
            number_of_layers,
            min_value=min_value,
            max_value=max_value,
            step=step,
        )

        # When
        build_uniform_param_grid(
            ansatz_specs=json.dumps(ansatz_specs),
            number_of_layers=number_of_layers,
            min_value=min_value,
            max_value=max_value,
            step=step,
        )

        # Then
        try:
            parameter_grid = load_parameter_grid(expected_parameter_grid_filename)
            assert [
                tuple(param) for param in parameter_grid.param_ranges
            ] == expected_parameter_grid.param_ranges
        finally:
            remove_file_if_exists(expected_parameter_grid_filename)

    @pytest.mark.parametrize(
        "number_of_params_per_layer, number_of_layers, min_value, max_value, step",
        [
            (0, 2, 0, 1, 0.5),
            (1, 2, 0, 1, 0.5),
            (6, 2, 0, 1, 0.5),
            (1, 6, 0, 1, 0.5),
            (1, 6, -np.pi, 1, 0.5),
            (1, 6, -np.pi, np.pi, 0.5),
            (1, 6, 0, 1, 0.01),
        ],
    )
    def test_build_uniform_param_grid_number_of_params_per_layer(
        self, number_of_params_per_layer, number_of_layers, min_value, max_value, step
    ):
        # Given
        expected_parameter_grid_filename = "parameter-grid.json"
        expected_parameter_grid = _build_uniform_param_grid(
            number_of_params_per_layer,
            number_of_layers,
            min_value=min_value,
            max_value=max_value,
            step=step,
        )

        # When
        build_uniform_param_grid(
            number_of_params_per_layer=number_of_params_per_layer,
            number_of_layers=number_of_layers,
            min_value=min_value,
            max_value=max_value,
            step=step,
        )

        # Then
        try:
            parameter_grid = load_parameter_grid(expected_parameter_grid_filename)
            assert [
                tuple(param) for param in parameter_grid.param_ranges
            ] == expected_parameter_grid.param_ranges
        finally:
            remove_file_if_exists(expected_parameter_grid_filename)

    def test_build_uniform_param_grid_fails_with_both_ansatz_specs_and_number_of_params_per_layer(
        self,
    ):
        expected_parameter_grid_filename = "parameter-grid.json"
        number_of_params_per_layer = 2
        ansatz_specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": "MockAnsatz",
            "number_of_layers": 2,
            "problem_size": 1,
        }

        try:
            with pytest.raises(AssertionError):
                build_uniform_param_grid(
                    ansatz_specs=ansatz_specs,
                    number_of_params_per_layer=number_of_params_per_layer,
                )
        finally:
            remove_file_if_exists(expected_parameter_grid_filename)

    def test_build_uniform_param_grid_fails_with_neither_ansatz_specs_nor_number_of_params_per_layer(
        self,
    ):
        expected_parameter_grid_filename = "parameter-grid.json"
        try:
            with pytest.raises(AssertionError):
                build_uniform_param_grid()
        finally:
            remove_file_if_exists(expected_parameter_grid_filename)


class TestBuildCircuitLayersAndConnectivity:
    @pytest.mark.parametrize(
        "x_dimension, y_dimension, layer_type",
        [
            (0, None, "nearest-neighbor"),
            (1, None, "nearest-neighbor"),
            (2, None, "nearest-neighbor"),
            (0, 0, "nearest-neighbor"),
            (1, 0, "nearest-neighbor"),
            (2, 0, "nearest-neighbor"),
            (0, 1, "nearest-neighbor"),
            (1, 1, "nearest-neighbor"),
            (2, 1, "nearest-neighbor"),
            (0, 2, "nearest-neighbor"),
            (1, 2, "nearest-neighbor"),
            (2, 2, "nearest-neighbor"),
            (1, 1, "sycamore"),
            (2, 1, "sycamore"),
            (1, 2, "sycamore"),
            (2, 2, "sycamore"),
        ],
    )
    def test_build_circuit_layers_and_connectivity(
        self, x_dimension, y_dimension, layer_type
    ):
        # Given
        expected_circuit_layers_filename = "circuit-layers.json"
        expected_circuit_connectivity_filename = "circuit-connectivity.json"
        (
            expected_circuit_connectivity,
            expected_circuit_layers,
        ) = _build_circuit_layers_and_connectivity(
            x_dimension=x_dimension, y_dimension=y_dimension, layer_type=layer_type
        )

        # When
        build_circuit_layers_and_connectivity(
            x_dimension=x_dimension, y_dimension=y_dimension, layer_type=layer_type
        )

        # Then
        try:
            circuit_layers = load_circuit_layers(expected_circuit_layers_filename)
            circuit_connectivity = load_circuit_connectivity(
                expected_circuit_connectivity_filename
            )
            assert circuit_layers.layers == expected_circuit_layers.layers
            assert (
                circuit_connectivity.connectivity
                == expected_circuit_connectivity.connectivity
            )
        finally:
            remove_file_if_exists(expected_circuit_connectivity_filename)
            remove_file_if_exists(expected_circuit_layers_filename)


class TestCreateRandomCircuit:
    @pytest.mark.parametrize(
        "number_of_qubits, number_of_gates, seed",
        [
            (2, 4, None),
            (2, 10, RNDSEED),
            (2, 100, RNDSEED),
            (2, 1000, RNDSEED),
            (2, 10000, RNDSEED),
            (5, 4, None),
            (5, 10, RNDSEED),
            (5, 100, RNDSEED),
            (5, 1000, RNDSEED),
            (5, 10000, RNDSEED),
            (17, 4, None),
            (17, 10, RNDSEED),
            (17, 100, RNDSEED),
            (17, 1000, RNDSEED),
            (17, 10000, RNDSEED),
            (35, 4, None),
            (35, 10, RNDSEED),
            (35, 100, RNDSEED),
            (35, 1000, RNDSEED),
            (35, 10000, RNDSEED),
        ],
    )
    def test_create_random_circuit(self, number_of_qubits, number_of_gates, seed):
        # Given
        expected_filename = "circuit.json"
        expected_circuit = _create_random_circuit(
            number_of_qubits,
            number_of_gates,
            seed=seed,
        )

        # When
        create_random_circuit(
            number_of_qubits=number_of_qubits,
            number_of_gates=number_of_gates,
            seed=seed,
        )

        # Then
        try:
            circuit = load_circuit(expected_filename)
            if seed is not None:
                assert circuit.gates == expected_circuit.gates
            else:
                assert circuit.gates != expected_circuit.gates
        finally:
            remove_file_if_exists(expected_filename)


class TestAddAncillaRegisterToCircuitPythonObject:
    @pytest.fixture(params=[i for i in range(1, 20, 3)])
    def number_of_ancilla_qubits(self, request):
        return request.param

    @pytest.fixture()
    def circuit_filename_and_number_of_ancilla_qubits(self, number_of_ancilla_qubits):
        number_of_qubits = 4
        number_of_gates = 10
        circuit = _create_random_circuit(
            number_of_qubits, number_of_gates, seed=RNDSEED
        )
        circuit_filename = "circuit.json"
        save_circuit(circuit, circuit_filename)

        yield circuit_filename, number_of_ancilla_qubits

        remove_file_if_exists(circuit_filename)

    def test_add_ancilla_register_to_circuit_python_object(
        self, number_of_ancilla_qubits
    ):
        # Given
        number_of_qubits = 4
        number_of_gates = 10
        circuit = _create_random_circuit(
            number_of_qubits, number_of_gates, seed=RNDSEED
        )
        expected_extended_cirucit = _add_ancilla_register_to_circuit(
            copy.deepcopy(circuit), number_of_ancilla_qubits
        )
        expected_extended_circuit_filename = "extended-circuit.json"

        # When
        add_ancilla_register_to_circuit(number_of_ancilla_qubits, circuit)

        # Then
        try:
            extended_circuit = load_circuit(expected_extended_circuit_filename)
            assert (
                len(extended_circuit.qubits)
                == number_of_qubits + number_of_ancilla_qubits
            )
            assert extended_circuit.gates == expected_extended_cirucit.gates
        finally:
            remove_file_if_exists(expected_extended_circuit_filename)

    def test_add_ancilla_register_to_circuit_artifact_file(
        self, circuit_filename_and_number_of_ancilla_qubits
    ):
        # Given
        (
            circuit_filename,
            number_of_ancilla_qubits,
        ) = circuit_filename_and_number_of_ancilla_qubits
        expected_extended_circuit_filename = "extended-circuit.json"

        circuit = load_circuit(circuit_filename)
        expected_extended_circuit = _add_ancilla_register_to_circuit(
            circuit, number_of_ancilla_qubits
        )

        # When
        add_ancilla_register_to_circuit(number_of_ancilla_qubits, circuit_filename)

        # Then
        try:
            extended_circuit = load_circuit(expected_extended_circuit_filename)
            assert (
                len(extended_circuit.qubits)
                == len(circuit.qubits) + number_of_ancilla_qubits
            )
            assert extended_circuit.gates == expected_extended_circuit.gates
        finally:
            remove_file_if_exists(expected_extended_circuit_filename)


class TestConcatenateCircuits:
    @pytest.fixture(params=[i for i in range(1, 20, 3)])
    def number_of_circuits(self, request):
        return request.param

    @pytest.fixture()
    def circuit_set(self, number_of_circuits):
        number_of_qubits = 4
        number_of_gates = 10
        circuit_set = [
            _create_random_circuit(number_of_qubits, number_of_gates, seed=RNDSEED)
            for _ in range(number_of_circuits)
        ]
        return circuit_set

    @pytest.fixture()
    def circuit_set_filename(self, circuit_set):
        circuit_set_filename = "circuit-set.json"
        save_circuit_set(circuit_set, circuit_set_filename)

        yield circuit_set_filename

        remove_file_if_exists(circuit_set_filename)

    def test_concatenate_circuits_python_objects(self, circuit_set):
        # Given
        expected_concatenated_circuit_filename = "result-circuit.json"
        expected_concatenated_circuit = Circuit()
        for circuit in copy.deepcopy(circuit_set):
            expected_concatenated_circuit += circuit

        # When
        concatenate_circuits(circuit_set)

        # Then
        try:
            concatenated_circuit = load_circuit(expected_concatenated_circuit_filename)
            assert concatenated_circuit.gates == expected_concatenated_circuit.gates
        finally:
            remove_file_if_exists(expected_concatenated_circuit_filename)

    def test_concatenate_circuits_artifact_file(self, circuit_set_filename):
        # Given
        expected_concatenated_circuit_filename = "result-circuit.json"

        circuit_set = load_circuit_set(circuit_set_filename)
        expected_concatenated_circuit = Circuit()
        for circuit in copy.deepcopy(circuit_set):
            expected_concatenated_circuit += circuit

        # When
        concatenate_circuits(circuit_set_filename)

        # Then
        try:
            concatenated_circuit = load_circuit(expected_concatenated_circuit_filename)
            assert concatenated_circuit.gates == expected_concatenated_circuit.gates
        finally:
            remove_file_if_exists(expected_concatenated_circuit_filename)


class TestBatchCircuits:
    @pytest.fixture(params=[0, 1, 4, 7])
    def input_circuits(self, request):
        number_of_qubits = 4
        number_of_gates = 10
        return [
            _create_random_circuit(number_of_qubits, number_of_gates, seed=RNDSEED + i)
            for i in range(request.param)
        ]

    @pytest.fixture()
    def input_circuits_filenames(self, input_circuits):
        circuit_filenames = []
        for i, circuit in enumerate(input_circuits):
            circuit_filenames.append(f"circuit-{i}.json")
            save_circuit(circuit, circuit_filenames[i])

        yield circuit_filenames

        for filename in circuit_filenames:
            remove_file_if_exists(filename)

    @pytest.fixture(params=[0, 3, 6, 8])
    def input_circuit_set(self, request):
        number_of_qubits = 4
        number_of_gates = 10
        return [
            _create_random_circuit(
                number_of_qubits, number_of_gates, seed=RNDSEED + 100 + i
            )
            for i in range(request.param)
        ]

    @pytest.fixture()
    def input_circuit_set_filename(self, input_circuit_set):
        circuit_set_filename = "input-circuit-set.json"
        save_circuit_set(input_circuit_set, circuit_set_filename)

        yield circuit_set_filename

        remove_file_if_exists(circuit_set_filename)

    def test_batch_circuits_all_artifacts_no_circuit_set(
        self, input_circuits_filenames
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuit_set = []
        for circuit_filename in input_circuits_filenames:
            expected_circuit_set.append(load_circuit(circuit_filename))

        # When
        batch_circuits(input_circuits_filenames)

        # Then
        try:
            circuit_set = load_circuit_set(expected_circuit_set_filename)
            assert circuit_set == expected_circuit_set
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_artifacts_circuit_set_is_artifact(
        self, input_circuits_filenames, input_circuit_set_filename
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuit_set = load_circuit_set(input_circuit_set_filename)
        for circuit_filename in input_circuits_filenames:
            expected_circuit_set.append(load_circuit(circuit_filename))

        # When
        batch_circuits(input_circuits_filenames, circuit_set=input_circuit_set_filename)

        # Then
        try:
            circuit_set = load_circuit_set(expected_circuit_set_filename)
            assert circuit_set == expected_circuit_set
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_artifacts_circuit_set_is_object(
        self, input_circuits_filenames, input_circuit_set
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuit_set = copy.deepcopy(input_circuit_set)
        for circuit_filename in input_circuits_filenames:
            expected_circuit_set.append(load_circuit(circuit_filename))

        # When
        batch_circuits(
            input_circuits_filenames, circuit_set=copy.deepcopy(input_circuit_set)
        )

        # Then
        try:
            circuit_set = load_circuit_set(expected_circuit_set_filename)
            assert circuit_set == expected_circuit_set
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_objects_no_circuit_set(self, input_circuits):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuit_set = copy.deepcopy(input_circuits)

        # When
        batch_circuits(input_circuits)

        # Then
        try:
            circuit_set = load_circuit_set(expected_circuit_set_filename)
            assert circuit_set == expected_circuit_set
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_objects_circuit_set_is_artifact(
        self, input_circuits, input_circuit_set_filename
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuit_set = load_circuit_set(input_circuit_set_filename)
        for circuit in input_circuits:
            expected_circuit_set.append(copy.deepcopy(circuit))

        # When
        batch_circuits(input_circuits, circuit_set=input_circuit_set_filename)

        # Then
        try:
            circuit_set = load_circuit_set(expected_circuit_set_filename)
            assert circuit_set == expected_circuit_set
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_objects_circuit_set_is_object(
        self, input_circuits, input_circuit_set
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuit_set = copy.deepcopy(input_circuit_set)
        for circuit in input_circuits:
            expected_circuit_set.append(copy.deepcopy(circuit))

        # When
        batch_circuits(input_circuits, circuit_set=copy.deepcopy(input_circuit_set))

        # Then
        try:
            circuit_set = load_circuit_set(expected_circuit_set_filename)
            assert circuit_set == expected_circuit_set
        finally:
            remove_file_if_exists(expected_circuit_set_filename)
