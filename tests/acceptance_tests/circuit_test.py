import copy
import json
import os

import numpy as np
import pytest
import zquantum.core.circuits as new_circuits
from zquantum.core.circuits.layouts import (
    build_circuit_layers_and_connectivity as _build_circuit_layers_and_connectivity,
)
from zquantum.core.circuits.layouts import (
    load_circuit_connectivity,
    load_circuit_layers,
)
from zquantum.core.serialization import load_array, save_array
from zquantum.core.utils import RNDSEED, create_object

from steps.circuit import (
    add_ancilla_register_to_circuit,
    batch_circuits,
    build_ansatz_circuit,
    build_circuit_layers_and_connectivity,
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
    def test_using_mock_ansatz_specs(self, number_of_layers):
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
            parameters = load_array(filename)
            assert len(parameters) == number_of_layers
        finally:
            remove_file_if_exists(filename)

    def test_ansatz_specs_as_string(self):
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
            parameters = load_array(filename)
            assert len(parameters) == number_of_layers
        finally:
            remove_file_if_exists(filename)

    @pytest.mark.parametrize(
        "number_of_parameters",
        [i for i in range(12)],
    )
    def test_using_number_of_parameters(
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
            parameters = load_array(filename)
            assert len(parameters) == number_of_parameters
        finally:
            remove_file_if_exists(filename)

    def test_fails_with_both_ansatz_specs_and_number_of_parameters(
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

    def test_fails_with_neither_ansatz_specs_nor_number_of_parameters(
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
        save_array(np.array(request.param[0]), params1_filename)

        params2_filename = "params2.json"
        save_array(np.array(request.param[1]), params2_filename)

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
            parameters = load_array(combined_parameters_filename)
            params1 = load_array(params1_filename)
            params2 = load_array(params2_filename)
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
        save_array(np.array(params), params_filename)

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

        parameters = load_array(params_filename)
        ansatz = create_object(copy.deepcopy(ansatz_specs))
        expected_circuit = ansatz.get_executable_circuit(parameters)

        # When
        build_ansatz_circuit(ansatz_specs=ansatz_specs, params=params_filename)

        # Then
        try:
            circuit_filename = "circuit.json"
            with open(circuit_filename) as f:
                circuit = new_circuits.circuit_from_dict(json.load(f))
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
            with open(circuit_filename) as f:
                circuit = new_circuits.circuit_from_dict(json.load(f))
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
            with open(circuit_filename) as f:
                circuit = new_circuits.circuit_from_dict(json.load(f))
            assert circuit == expected_circuit
        finally:
            remove_file_if_exists(circuit_filename)

    def test_build_ansatz_circuit_raises_exception_on_invalid_inputs(self):
        params_filename = "params.json"
        save_array(np.array([1.0]), params_filename)

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
        expected_circuit = new_circuits.create_random_circuit(
            number_of_qubits,
            number_of_gates,
            rng=np.random.default_rng(seed),
        )

        # When
        create_random_circuit(
            number_of_qubits=number_of_qubits,
            number_of_gates=number_of_gates,
            seed=seed,
        )

        # Then
        try:
            with open(expected_filename) as f:
                circuit = new_circuits.circuit_from_dict(json.load(f))
            if seed is not None:
                assert circuit.operations == expected_circuit.operations
            else:
                assert circuit.operations != expected_circuit.operations
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
        circuit = new_circuits.create_random_circuit(
            number_of_qubits,
            number_of_gates,
            rng=np.random.default_rng(RNDSEED),
        )
        circuit_filename = "circuit.json"
        with open(circuit_filename, "w") as f:
            json.dump(new_circuits.to_dict(circuit), f)

        yield circuit_filename, number_of_ancilla_qubits

        remove_file_if_exists(circuit_filename)

    def test_add_ancilla_register_to_circuit_python_object(
        self, number_of_ancilla_qubits
    ):
        # Given
        number_of_qubits = 4
        number_of_gates = 10
        circuit = new_circuits.create_random_circuit(
            number_of_qubits,
            number_of_gates,
            rng=np.random.default_rng(RNDSEED),
        )
        expected_extended_cirucit = new_circuits.add_ancilla_register(
            copy.deepcopy(circuit), number_of_ancilla_qubits
        )
        expected_extended_circuit_filename = "extended-circuit.json"

        # When
        circuit = add_ancilla_register_to_circuit(number_of_ancilla_qubits, circuit)

        # Then
        try:
            with open(expected_extended_circuit_filename) as f:
                extended_circuit = new_circuits.circuit_from_dict(json.load(f))
            assert (
                extended_circuit.n_qubits == number_of_qubits + number_of_ancilla_qubits
            )
            assert extended_circuit == expected_extended_cirucit
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

        with open(circuit_filename) as f:
            circuit = new_circuits.circuit_from_dict(json.load(f))
        expected_extended_circuit = new_circuits.add_ancilla_register(
            circuit, number_of_ancilla_qubits
        )

        # When
        add_ancilla_register_to_circuit(number_of_ancilla_qubits, circuit_filename)

        # Then
        try:
            with open(expected_extended_circuit_filename) as f:
                extended_circuit = new_circuits.circuit_from_dict(json.load(f))
            assert (
                extended_circuit.n_qubits == circuit.n_qubits + number_of_ancilla_qubits
            )
            assert extended_circuit == expected_extended_circuit
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
            new_circuits.create_random_circuit(
                number_of_qubits, number_of_gates, rng=np.random.default_rng(RNDSEED)
            )
            for _ in range(number_of_circuits)
        ]
        return circuit_set

    @pytest.fixture()
    def circuit_set_filename(self, circuit_set):
        circuit_set_filename = "circuit-set.json"
        with open(circuit_set_filename, "w") as f:
            json.dump(new_circuits.to_dict(circuit_set), f)

        yield circuit_set_filename

        remove_file_if_exists(circuit_set_filename)

    def test_concatenate_circuits_python_objects(self, circuit_set):
        # Given
        expected_concatenated_circuit_filename = "result-circuit.json"
        expected_concatenated_circuit = sum(
            [circuit for circuit in circuit_set], new_circuits.Circuit()
        )

        # When
        concatenate_circuits(circuit_set)

        # Then
        try:
            with open(expected_concatenated_circuit_filename) as f:
                concatenated_circuit = new_circuits.circuit_from_dict(json.load(f))
            assert concatenated_circuit == expected_concatenated_circuit
        finally:
            remove_file_if_exists(expected_concatenated_circuit_filename)

    def test_concatenate_circuits_artifact_file(self, circuit_set_filename):
        # Given
        expected_concatenated_circuit_filename = "result-circuit.json"

        with open(circuit_set_filename) as f:
            circuit_set = new_circuits.circuitset_from_dict(json.load(f))
        expected_concatenated_circuit = sum(
            [circuit for circuit in circuit_set], new_circuits.Circuit()
        )
        # When
        concatenate_circuits(circuit_set_filename)

        # Then
        try:
            with open(expected_concatenated_circuit_filename) as f:
                concatenated_circuit = new_circuits.circuit_from_dict(json.load(f))
            assert concatenated_circuit == expected_concatenated_circuit
        finally:
            remove_file_if_exists(expected_concatenated_circuit_filename)


class TestBatchCircuits:
    @pytest.fixture(params=[0, 1, 4, 7])
    def input_circuits(self, request):
        number_of_qubits = 4
        number_of_gates = 10
        rng = np.random.default_rng(RNDSEED)
        return [
            new_circuits.create_random_circuit(
                number_of_qubits, number_of_gates, rng=rng
            )
            for _ in range(request.param)
        ]

    @pytest.fixture()
    def input_circuits_filenames(self, input_circuits):
        circuit_filenames = [f"circuit-{i}.json" for i in range(len(input_circuits))]
        for circuit, filename in zip(input_circuits, circuit_filenames):
            with open(filename, "w") as f:
                json.dump(new_circuits.to_dict(circuit), f)

        yield circuit_filenames

        for filename in circuit_filenames:
            remove_file_if_exists(filename)

    @pytest.fixture(params=[0, 3, 6, 8])
    def input_circuit_set(self, request):
        number_of_qubits = 4
        number_of_gates = 10
        rng = np.random.default_rng(RNDSEED + 100)
        return [
            new_circuits.create_random_circuit(
                number_of_qubits, number_of_gates, rng=rng
            )
            for _ in range(request.param)
        ]

    @pytest.fixture()
    def input_circuit_set_filename(self, input_circuit_set):
        filename = "input-circuit-set.json"
        with open(filename, "w") as f:
            json.dump(new_circuits.to_dict(input_circuit_set), f)

        yield filename

        remove_file_if_exists(filename)

    def test_batch_circuits_all_artifacts_no_circuit_set(
        self, input_circuits_filenames
    ):
        # Given
        expected_circuitset_filename = "circuit-set.json"
        expected_circuitset = []
        for circuit_filename in input_circuits_filenames:
            with open(circuit_filename) as f:
                circuit = new_circuits.circuit_from_dict(json.load(f))
            expected_circuitset.append(circuit)

        # When
        batch_circuits(input_circuits_filenames)

        # Then
        try:
            with open(expected_circuitset_filename) as f:
                circuitset = new_circuits.circuitset_from_dict(json.load(f))
            assert circuitset == expected_circuitset
        finally:
            remove_file_if_exists(expected_circuitset_filename)

    def test_batch_circuits_all_artifacts_circuit_set_is_artifact(
        self, input_circuits_filenames, input_circuit_set_filename
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        with open(input_circuit_set_filename) as f:
            expected_circuitset = new_circuits.circuitset_from_dict(json.load(f))
        for circuit_filename in input_circuits_filenames:
            with open(circuit_filename) as f:
                expected_circuitset.append(new_circuits.circuit_from_dict(json.load(f)))

        # When
        batch_circuits(input_circuits_filenames, circuit_set=input_circuit_set_filename)

        # Then
        try:
            with open(expected_circuit_set_filename) as f:
                circuitset = new_circuits.circuitset_from_dict(json.load(f))
            assert circuitset == expected_circuitset
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_artifacts_circuit_set_is_object(
        self, input_circuits_filenames, input_circuit_set
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuitset = copy.deepcopy(input_circuit_set)
        for circuit_filename in input_circuits_filenames:
            with open(circuit_filename) as f:
                circuit = new_circuits.circuit_from_dict(json.load(f))
            expected_circuitset.append(circuit)

        # When
        batch_circuits(
            input_circuits_filenames, circuit_set=copy.deepcopy(input_circuit_set)
        )

        # Then
        try:
            with open(expected_circuit_set_filename) as f:
                circuitset = new_circuits.circuitset_from_dict(json.load(f))
            assert circuitset == expected_circuitset
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_objects_no_circuit_set(self, input_circuits):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuitset = copy.deepcopy(input_circuits)

        # When
        batch_circuits(input_circuits)

        # Then
        try:
            with open(expected_circuit_set_filename) as f:
                circuitset = new_circuits.circuitset_from_dict(json.load(f))
            assert circuitset == expected_circuitset
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_objects_circuit_set_is_artifact(
        self, input_circuits, input_circuit_set_filename
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        with open(input_circuit_set_filename) as f:
            expected_circuitset = new_circuits.circuitset_from_dict(json.load(f))
        for circuit in input_circuits:
            expected_circuitset.append(copy.deepcopy(circuit))

        # When
        batch_circuits(input_circuits, circuit_set=input_circuit_set_filename)

        # Then
        try:
            with open(expected_circuit_set_filename) as f:
                circuitset = new_circuits.circuitset_from_dict(json.load(f))
            assert circuitset == expected_circuitset
        finally:
            remove_file_if_exists(expected_circuit_set_filename)

    def test_batch_circuits_all_objects_circuit_set_is_object(
        self, input_circuits, input_circuit_set
    ):
        # Given
        expected_circuit_set_filename = "circuit-set.json"
        expected_circuitset = copy.deepcopy(input_circuit_set)
        for circuit in input_circuits:
            expected_circuitset.append(copy.deepcopy(circuit))

        # When
        batch_circuits(input_circuits, circuit_set=copy.deepcopy(input_circuit_set))

        # Then
        try:
            with open(expected_circuit_set_filename) as f:
                circuitset = new_circuits.circuitset_from_dict(json.load(f))
            assert circuitset == expected_circuitset
        finally:
            remove_file_if_exists(expected_circuit_set_filename)
