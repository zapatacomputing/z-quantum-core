import json
from typing import List, Optional, Union

import numpy as np
import numpy.random
import zquantum.core.wip.circuits as new_circuits
from zquantum.core.circuit import Circuit
from zquantum.core.circuit import (
    build_circuit_layers_and_connectivity as _build_circuit_layers_and_connectivity,
)
from zquantum.core.circuit import build_uniform_param_grid as _build_uniform_param_grid
from zquantum.core.circuit import combine_ansatz_params as _combine_ansatz_params
from zquantum.core.circuit import (
    load_circuit,
    load_circuit_set,
    load_circuit_template_params,
    save_circuit,
    save_circuit_connectivity,
    save_circuit_layers,
    save_circuit_set,
    save_circuit_template_params,
    save_parameter_grid,
)
from zquantum.core.typing import Specs
from zquantum.core.utils import create_symbols_map, load_from_specs


# Generate random parameters for an ansatz
def generate_random_ansatz_params(
    ansatz_specs: Optional[Specs] = None,
    number_of_parameters: Optional[int] = None,
    min_value: float = -np.pi * 0.5,
    max_value: float = np.pi * 0.5,
    seed: Optional[int] = None,
):
    assert (ansatz_specs is None) != (number_of_parameters is None)

    if ansatz_specs is not None:
        ansatz = load_from_specs(ansatz_specs)
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
def build_ansatz_circuit(
    ansatz_specs: Specs, params: Optional[Union[str, List]] = None
):
    ansatz = load_from_specs(ansatz_specs)
    if params is not None:
        if isinstance(params, str):
            params = load_circuit_template_params(params)
        else:
            params = np.array(params)
        circuit = ansatz.get_executable_circuit(params)
    elif ansatz.supports_parametrized_circuits:
        circuit = ansatz.parametrized_circuit
    else:
        raise (
            Exception(
                "Ansatz is not parametrizable and no parameters has been provided."
            )
        )
    with open("circuit.json", "w") as f:
        json.dump(new_circuits.to_dict(circuit), f)


# Build uniform parameter grid
def build_uniform_param_grid(
    ansatz_specs: Optional[Specs] = None,
    number_of_params_per_layer: Optional[int] = None,
    number_of_layers: int = 1,
    min_value: float = 0,
    max_value: float = 2 * np.pi,
    step: float = np.pi / 5,
):
    assert (ansatz_specs is None) != (number_of_params_per_layer is None)

    if ansatz_specs is not None:
        ansatz = load_from_specs(ansatz_specs)
        number_of_params = ansatz.number_of_params
    else:
        number_of_params = number_of_params_per_layer

    grid = _build_uniform_param_grid(
        number_of_params, number_of_layers, min_value, max_value, step
    )
    save_parameter_grid(grid, "parameter-grid.json")


# Build circuit layers and connectivity
def build_circuit_layers_and_connectivity(
    x_dimension: int,
    y_dimension: Optional[int] = None,
    layer_type: str = "nearest-neighbor",
):
    connectivity, layers = _build_circuit_layers_and_connectivity(
        x_dimension, y_dimension, layer_type
    )
    save_circuit_layers(layers, "circuit-layers.json")
    save_circuit_connectivity(connectivity, "circuit-connectivity.json")


# Create random circuit
def create_random_circuit(
    number_of_qubits: int, number_of_gates: int, seed: Optional[int] = None
):
    rng = np.random.default_rng(seed)
    circuit = new_circuits.create_random_circuit(
        number_of_qubits, number_of_gates, rng=rng
    )
    with open("circuit.json", "w") as f:
        json.dump(new_circuits.to_dict(circuit), f)


# Add register of ancilla qubits to circuit
def add_ancilla_register_to_circuit(
    number_of_ancilla_qubits: int, circuit: Union[Circuit, str]
):
    if isinstance(circuit, str):
        with open(circuit) as f:
            circuit = new_circuits.circuit_from_dict(json.load(f))
    extended_circuit = new_circuits.add_ancilla_register(
        circuit, number_of_ancilla_qubits
    )
    with open("extended-circuit.json", "w") as f:
        json.dump(new_circuits.to_dict(extended_circuit), f)


# Concatenate circuits in a circuitset to create a composite circuit
def concatenate_circuits(circuit_set: Union[str, List[Circuit]]):
    if isinstance(circuit_set, str):
        with open(circuit_set) as f:
            circuit_set = new_circuits.circuit_from_dict(json.load(f))
    result_circuit = sum(circuit_set, new_circuits.Circuit())
    with open("result-circuit.json", "w") as f:
        json.dump(new_circuits.to_dict(result_circuit), f)


# Create one circuitset from circuit and circuitset objects
def batch_circuits(
    circuits: List[Union[str, Circuit]],
    circuit_set: Optional[Union[str, List[Circuit]]] = None,
):
    if circuit_set is None:
        circuit_set = []
    else:
        if isinstance(circuit_set, str):
            circuit_set = load_circuit_set(circuit_set)

    for circuit in circuits:
        if isinstance(circuit, str):
            circuit = load_circuit(circuit)
        circuit_set.append(circuit)

    save_circuit_set(circuit_set, "circuit-set.json")


def evaluate_parametrized_circuit(
    parametrized_circuit: Union[str, new_circuits.Circuit],
    parameters: Union[str, np.ndarray],
):
    if isinstance(parametrized_circuit, str):
        with open(parametrized_circuit) as f:
            parametrized_circuit = new_circuits.circuit_from_dict(json.load(f))

    if isinstance(parameters, str):
        parameters = load_circuit_template_params(parameters)

    symbols_map = create_symbols_map(parametrized_circuit.symbolic_params, parameters)
    bound_circuit = parametrized_circuit.bind(symbols_map)
    with open("evaluated-circuit.json", "w") as f:
        json.dump(new_circuits.to_dict(bound_circuit), f)
