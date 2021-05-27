import json
from typing import List, Optional, Union

import numpy as np
import numpy.random
import zquantum.core.wip.circuits as new_circuits
import zquantum.core.wip.circuits.layouts as layouts
from zquantum.core import serialization
from zquantum.core.circuit import combine_ansatz_params as _combine_ansatz_params
from zquantum.core.typing import Specs
from zquantum.core.utils import create_symbols_map, load_from_specs
from zquantum.core.wip.circuits import Circuit


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
    layouts.save_circuit_template_params(params, "params.json")


# Combine two sets of ansatz parameters
def combine_ansatz_params(params1: str, params2: str):
    parameters1 = serialization.load_array(params1)
    parameters2 = serialization.load_array(params2)
    combined_params = _combine_ansatz_params(parameters1, parameters2)
    layouts.save_circuit_template_params(combined_params, "combined-params.json")


# Build circuit from ansatz
def build_ansatz_circuit(
    ansatz_specs: Specs, params: Optional[Union[str, List]] = None
):
    ansatz = load_from_specs(ansatz_specs)
    params_array: np.ndarray
    if params is not None:
        if isinstance(params, str):
            params_array = serialization.load_array(params)
        else:
            params_array = np.array(params)
        circuit = ansatz.get_executable_circuit(params_array)
    elif ansatz.supports_parametrized_circuits:
        circuit = ansatz.parametrized_circuit
    else:
        raise (
            Exception(
                "Ansatz is not parametrizable and no parameters has been provided."
            )
        )
    new_circuits.save_circuit(circuit, "circuit.json")


# Build circuit layers and connectivity
def build_circuit_layers_and_connectivity(
    x_dimension: int,
    y_dimension: Optional[int] = None,
    layer_type: str = "nearest-neighbor",
):
    connectivity, layers = layouts.build_circuit_layers_and_connectivity(
        x_dimension, y_dimension, layer_type
    )
    layouts.save_circuit_layers(layers, "circuit-layers.json")
    layouts.save_circuit_connectivity(connectivity, "circuit-connectivity.json")


# Create random circuit
def create_random_circuit(
    number_of_qubits: int, number_of_gates: int, seed: Optional[int] = None
):
    rng = np.random.default_rng(seed)
    circuit = new_circuits.create_random_circuit(
        number_of_qubits, number_of_gates, rng=rng
    )
    new_circuits.save_circuit(circuit, "circuit.json")


# Add register of ancilla qubits to circuit
def add_ancilla_register_to_circuit(
    number_of_ancilla_qubits: int, circuit: Union[Circuit, str]
):
    if isinstance(circuit, str):
        circuit = new_circuits.load_circuit(circuit)
    extended_circuit = new_circuits.add_ancilla_register(
        circuit, number_of_ancilla_qubits
    )
    new_circuits.save_circuit("extended-circuit.json", extended_circuit)


# Concatenate circuits in a circuitset to create a composite circuit
def concatenate_circuits(circuit_set: Union[str, List[Circuit]]):
    if isinstance(circuit_set, str):
        circuit_set = new_circuits.load_circuit_set(circuit_set)
    result_circuit = sum(circuit_set, new_circuits.Circuit())
    new_circuits.save_circuitset(result_circuit)


# Create one circuitset from circuit and circuitset objects
def batch_circuits(
    circuits: List[Union[str, Circuit]],
    circuit_set: Optional[Union[str, List[Circuit]]] = None,
):
    loaded_circuit_set: List[Circuit]
    if circuit_set is None:
        loaded_circuit_set = []
    elif isinstance(circuit_set, str):
        loaded_circuit_set = new_circuits.load_circuitset(circuit_set)
    else:
        loaded_circuit_set = circuit_set

    for circuit in circuits:
        if isinstance(circuit, str):
            loaded_circuit = new_circuits.load_circuits(circuit)
        else:
            loaded_circuit = circuit

        loaded_circuit_set.append(loaded_circuit)

    with open("circuit-set.json", "w") as f:
        json.dump(new_circuits.to_dict(loaded_circuit_set), f)


def evaluate_parametrized_circuit(
    parametrized_circuit: Union[str, new_circuits.Circuit],
    parameters: Union[str, np.ndarray],
):
    if isinstance(parametrized_circuit, str):
        parametrized_circuit = new_circuits.load_circuit(parametrized_circuit)

    if isinstance(parameters, str):
        parameters = serialization.load_array(parameters)

    symbols_map = create_symbols_map(parametrized_circuit.symbolic_params, parameters)
    bound_circuit = parametrized_circuit.bind(symbols_map)
    new_circuits.save_circuit(bound_circuit, "evaluated-circuit.json")
