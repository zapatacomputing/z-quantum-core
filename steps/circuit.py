import numpy as np
import json
from zquantum.core.circuit import (
    build_circuit_layers_and_connectivity as _build_circuit_layers_and_connectivity,
    add_ancilla_register_to_circuit as _add_ancilla_register_to_circuit,
    combine_ansatz_params as _combine_ansatz_params,
    build_uniform_param_grid as _build_uniform_param_grid,
    save_circuit_template_params,
    load_circuit_template_params,
    save_circuit,
    save_parameter_grid,
    save_circuit_layers,
    save_circuit_connectivity,
    load_circuit,
    load_circuit_set,
    Circuit,
    save_circuit_set,
)
from zquantum.core.utils import create_object
from zquantum.core.testing import create_random_circuit as _create_random_circuit

from typing import Dict, Union

# Generate random parameters for an ansatz
def generate_random_ansatz_params(
    ansatz_specs: Union[str, Dict] = None,
    number_of_parameters: int = None,
    min_value: float = -np.pi * 0.5,
    max_value: float = np.pi * 0.5,
    seed: int = None,
):
    assert (ansatz_specs is None) or (number_of_parameters is None)
    assert not ((ansatz_specs is None) and (number_of_parameters is None))

    if ansatz_specs is not None:
        if isinstance(ansatz_specs, str):
            ansatz_specs_dict = json.loads(ansatz_specs)
        else:
            ansatz_specs_dict = ansatz_specs
        ansatz = create_object(ansatz_specs_dict)
        number_of_parameters = ansatz.number_of_params

    if seed is not None:
        np.random.seed(seed)

    params = np.random.uniform(min_value, max_value, number_of_parameters)
    save_circuit_template_params(params, "params.json")


# Combine two sets of ansatz parameters
def combine_ansatz_params(params1: str, params2: str):
    parameters1 = load_circuit_template_params(params1)
    parameters2 = load_circuit_template_params(params2)
    combined_params = _combine_ansatz_params(parameters1, parameters2)
    save_circuit_template_params(combined_params, "combined-params.json")


# Build circuit from ansatz
def build_ansatz_circuit(ansatz_specs: Union[str, Dict], params: str = None):
    if isinstance(ansatz_specs, str):
        ansatz_specs = json.loads(ansatz_specs)
    ansatz = create_object(ansatz_specs)
    if params is not None:
        parameters = load_circuit_template_params(params)
        circuit = ansatz.get_executable_circuit(parameters)
    elif ansatz.supports_parametrized_circuits:
        circuit = ansatz.parametrized_circuit
    else:
        raise (
            Exception(
                "Ansatz is not parametrizable and no parameters has been provided."
            )
        )
    save_circuit(circuit, "circuit.json")


# Build uniform parameter grid
def build_uniform_param_grid(
    ansatz_specs: Union[str, Dict] = None,
    number_of_params_per_layer: int = None,
    number_of_layers: int = 1,
    min_value: float = 0,
    max_value: float = 2 * np.pi,
    step: float = np.pi / 5,
):
    assert (ansatz_specs is None) or (number_of_params_per_layer is None)
    assert not ((ansatz_specs is None) and (number_of_params_per_layer is None))

    if ansatz_specs is not None:
        if isinstance(ansatz_specs, str):
            ansatz_specs = json.loads(ansatz_specs)
        ansatz = create_object(ansatz_specs)
        number_of_params = ansatz.number_of_params
    elif number_of_params_per_layer != "None":
        number_of_params = number_of_params_per_layer

    grid = _build_uniform_param_grid(
        number_of_params, number_of_layers, min_value, max_value, step
    )
    save_parameter_grid(grid, "parameter-grid.json")


# Build circuit layers and connectivity
def build_circuit_layers_and_connectivity(
    x_dimension: int,
    y_dimension: int = None,
    layer_type: str = "nearest-neighbor",
):
    connectivity, layers = _build_circuit_layers_and_connectivity(
        x_dimension, y_dimension, layer_type
    )
    save_circuit_layers(layers, "circuit-layers.json")
    save_circuit_connectivity(connectivity, "circuit-connectivity.json")


# Create random circuit
def create_random_circuit(
    number_of_qubits: int, number_of_gates: int, seed: Union[str, int] = "None"
):
    circuit = _create_random_circuit(number_of_qubits, number_of_gates, seed=seed)
    save_circuit(circuit, "circuit.json")


# Add register of ancilla qubits to circuit
def add_ancilla_register_to_circuit(number_of_ancilla_qubits: int, circuit: str):
    circuit_object = load_circuit(circuit)
    extended_circuit = _add_ancilla_register_to_circuit(
        circuit_object, number_of_ancilla_qubits
    )
    save_circuit(extended_circuit, "extended-circuit.json")


# Concatenate circuits in a circuitset to create a composite circuit
def concatenate_circuits(circuit_set: str):
    circuit_set_object = load_circuit_set(circuit_set)
    result_circuit = Circuit()
    for circuit in circuit_set_object:
        result_circuit += circuit
    save_circuit(result_circuit, "result-circuit.json")


# Create circuitset from circuit artifacts
def create_circuit_set_from_circuit_artifacts(
    circuit1: str,
    circuit2: str = "None",
    circuit3: str = "None",
    circuit4: str = "None",
    circuit_set: str = "None",
):
    if circuit_set != "None":  # TODO None isse in workflow v1
        circuit_set_object = load_circuit_set(circuit_set)
    else:
        circuit_set_object = []

    object_names = [circuit1, circuit2, circuit3, circuit4]
    for object in object_names:
        if object != "None":
            circuit_set_object.append(load_circuit(object))

    save_circuit_set(circuit_set_object, "circuit-set.json")
