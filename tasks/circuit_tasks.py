import numpy as np
from json import loads
import os
from zquantum.core.circuit import (
    save_circuit_template_params,
    combine_ansatz_params,
    load_circuit_template_params,
    save_circuit,
    save_parameter_grid,
    build_uniform_param_grid,
)
from zquantum.core.utils import create_object

# Generate random parameters for an ansatz
def generate_random_ansatz_params(
    ansatz_specs,
    # number_of_parameters=None,
    min_value=-np.pi * 0.5,
    max_value=np.pi * 0.5,
    seed=None,
):
    if ansatz_specs is not None:
        ansatz_specs_dict = loads(ansatz_specs)
        ansatz = create_object(ansatz_specs_dict)
        number_of_params = ansatz.number_of_params
    # elif number_of_parameters is not None:
    #     number_of_params = number_of_parameters
    if seed is not None:
        np.random.seed(seed)
    params = np.random.uniform(min_value, max_value, number_of_params)
    save_circuit_template_params(params, "params.json")


# Combine two sets of ansatz parameters
def combine_ansatz_params(params_1, params_2):
    params1 = load_circuit_template_params(loads(params_1))
    params2 = load_circuit_template_params(loads(params_2))
    combined_params = combine_ansatz_params(params1, params2)
    save_circuit_template_params(combined_params, "combined_params.json")


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
    ansatz_specs=None,
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
