from typing import List, Optional, Union

import numpy as np
import zquantum.core.circuits.layouts as layouts
from zquantum.core import circuits, serialization
from zquantum.core.circuits import (
    Circuit,
    load_circuit,
    load_circuitset,
    save_circuit,
    save_circuitset,
)
from zquantum.core.interfaces import ansatz_utils
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
    serialization.save_array(params, "params.json")


# Combine two sets of ansatz parameters
def combine_ansatz_params(params1: str, params2: str):
    parameters1 = serialization.load_array(params1)
    parameters2 = serialization.load_array(params2)
    combined_params = ansatz_utils.combine_ansatz_params(parameters1, parameters2)
    serialization.save_array(combined_params, "combined-params.json")


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
    save_circuit(circuit, "circuit.json")


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
    circuit = circuits.create_random_circuit(number_of_qubits, number_of_gates, rng=rng)
    save_circuit(circuit, "circuit.json")


# Add register of ancilla qubits to circuit
def add_ancilla_register_to_circuit(
    number_of_ancilla_qubits: int, circuit: Union[Circuit, str]
):
    if isinstance(circuit, str):
        circuit = load_circuit(circuit)
    extended_circuit = circuits.add_ancilla_register(circuit, number_of_ancilla_qubits)
    save_circuit(extended_circuit, "extended-circuit.json")


# Concatenate circuits in a circuitset to create a composite circuit
def concatenate_circuits(circuit_set: Union[str, List[Circuit]]):
    if isinstance(circuit_set, str):
        circuit_set = load_circuitset(circuit_set)
    result_circuit = sum(circuit_set, Circuit())
    save_circuitset(result_circuit, "result-circuit.json")


# Create one circuitset from circuit and circuitset objects
def batch_circuits(
    circuits: List[Union[str, Circuit]],
    circuit_set: Optional[Union[str, List[Circuit]]] = None,
):
    loaded_circuit_set: List[Circuit]
    if circuit_set is None:
        loaded_circuit_set = []
    elif isinstance(circuit_set, str):
        loaded_circuit_set = load_circuitset(circuit_set)
    else:
        loaded_circuit_set = circuit_set

    for circuit in circuits:
        if isinstance(circuit, str):
            loaded_circuit = load_circuit(circuit)
        else:
            loaded_circuit = circuit

        loaded_circuit_set.append(loaded_circuit)

    save_circuitset(loaded_circuit_set, "circuit-set.json")


def evaluate_parametrized_circuit(
    parametrized_circuit: Union[str, Circuit],
    parameters: Union[str, np.ndarray],
):
    if isinstance(parametrized_circuit, str):
        parametrized_circuit = load_circuit(parametrized_circuit)

    if isinstance(parameters, str):
        parameters = serialization.load_array(parameters)

    symbols_map = create_symbols_map(parametrized_circuit.symbolic_params, parameters)
    bound_circuit = parametrized_circuit.bind(symbols_map)
    save_circuit(bound_circuit, "evaluated-circuit.json")
