import numpy as np
from json import loads
from zquantum.core.circuit import (
    save_circuit_template_params,
    combine_ansatz_params,
    load_circuit_template_params,
    save_circuit,
    save_parameter_grid,
    build_uniform_param_grid,
    save_circuit_layers,
    save_circuit_connectivity,
    build_circuit_layers_and_connectivity,
    load_circuit,
    add_ancilla_register_to_circuit,
    load_circuit_set,
    Circuit,
    save_circuit_set,
)
from zquantum.core.utils import create_object
from zquantum.core.testing import create_random_circuit

# Generate random parameters for an ansatz
def generate_random_ansatz_params(
    ansatz_specs,
    number_of_parameters="None",
    min_value=-np.pi * 0.5,
    max_value=np.pi * 0.5,
    seed="None",
):
    print(f"Start", flush=True)
    print(f"type ansatz specs: {type(ansatz_specs)}", flush=True)
    if ansatz_specs is not "None":
        print(f"aaa", flush=True)
        ansatz_specs_dict = loads(ansatz_specs)
        ansatz = create_object(ansatz_specs_dict)
        number_of_params = ansatz.number_of_params
    elif number_of_parameters is not "None":
        print(f"bbb", flush=True)
        number_of_params = number_of_parameters
    if seed is not "None":
        print(f"seed", flush=True)
        np.random.seed(seed)
    print("number_of_parameters:", number_of_parameters)
    print("min_value:", min_value)
    print("max_value:", max_value)
    print("seed", seed)
    params = np.random.uniform(min_value, max_value, number_of_params)
    save_circuit_template_params(params, "params.json")


# Combine two sets of ansatz parameters
def combine_ansatz_params(params1, params2):
    params1_dict = loads(params1)
    params2_dict = loads(params2)
    parameters1 = load_circuit_template_params(params1_dict)
    parameters2 = load_circuit_template_params(params2_dict)
    combined_params = combine_ansatz_params(parameters1, parameters2)
    save_circuit_template_params(combined_params, "combined-params.json")


# Build circuit from ansatz
def build_ansatz_circuit(ansatz_specs, params=None):
    ansatz = create_object(loads(ansatz_specs))
    if params is not None:
        parameters = load_circuit_template_params(params)
        circuit = ansatz.get_executable_circuit(parameters)
    elif ansatz.support_parametrized_circuits:
        circuit = ansatz.parametrized_circuit
    else:
        raise (
            Exception(
                "Ansatz is not parametrizable and no parameters has been provided."
            )
        )
    save_circuit(circuit, "circuit.json")


# Build uniform parameter grid
def build_uniform_parameter_grid(
    ansatz_specs,
    number_of_params_per_layers=None,
    number_of_layers=1,
    min_value=0,
    max_value=2 * np.pi,
    step=np.pi / 5,
):
    if ansatz_specs is not None:
        ansatz = create_object(loads(ansatz_specs))
        number_of_params = ansatz.number_of_params
    elif number_of_params_per_layers is not None:
        number_of_params = number_of_params_per_layers

    grid = build_uniform_param_grid(
        number_of_params, number_of_layers, min_value, max_value, step
    )
    save_parameter_grid(grid, "parameter_grid.json")


# Build circuit layers and connectivity
def build_circuit_layers_and_connectivity(
    x_dimension, y_dimension="None", layer_type="nearest-neighbor"
):
    connectivity, layers = build_circuit_layers_and_connectivity(
        x_dimension, y_dimension, layer_type
    )
    save_circuit_layers(layers, "circuit-layers.json")
    save_circuit_connectivity(connectivity, "circuit-connectivity.json")


# Create random circuit
def create_random_circuit(number_of_qubits, number_of_gates, seed=None):
    circuit = create_random_circuit(number_of_qubits, number_of_gates, seed=seed)
    save_circuit(circuit, "circuit.json")


# Add register of ancilla qubits to circuit
def add_ancilla_qubits_register_to_circuit(number_of_ancilla_qubits, circuit):
    circuit_object = load_circuit(circuit)
    extended_circuit = add_ancilla_qubits_register_to_circuit(
        circuit, number_of_ancilla_qubits
    )
    save_circuit(extended_circuit, "extended_circuit.json")


# Concatenate circuits in a circuitset to create a composite circuit
def concatenate_circuits(circuit_set):
    circuit_set_object = load_circuit_set(circuit_set)
    result_circuit = Circuit()
    for circuit in circuit_set_object:
        result_circuit += circuit
    save_circuit(result_circuit, "result_circuit.json")


# Create circuitset from circuit artifacts
def create_circuit_set_from_circuit_artifacts(
    circuit1, circuit2=None, circuit3=None, circuit4=None, circuit_set=None
):
    if circuit_set is not None:
        circuit_set_object = load_circuit_set(circuit_set)
    else:
        circuit_set_object = []

    object_names = [circuit1, circuit2, circuit3, circuit4]
    for object in object_names:
        if object is not None:
            circuit_set_object.append(load_circuit(object))

    save_circuit_set(circuit_set, "circuit_set.json")
