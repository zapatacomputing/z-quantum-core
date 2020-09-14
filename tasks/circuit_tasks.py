import numpy as np
from json import loads
import os
from zquantum.core.circuit import (
    save_circuit_template_params,
    combine_ansatz_params,
    load_circuit_template_params,
    save_circuit,
)
from zquantum.core.utils import create_object

# Generate random parameters for an ansatz
def generate_random_ansatz_params(ansatz_specs, min_value, max_value, seed=None):
    if ansatz_specs is not None:
        ansatz_specs_dict = loads(ansatz_specs)
        ansatz = create_object(ansatz_specs_dict)
        number_of_params = ansatz.number_of_params
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
        parameters = load_circuit_template_params(loads(params))
        circuit = ansatz.get_executable_circuit(paramters)
    elif ansatz.support_parametrized_circuits:
        circuit = ansatz.parametrized_circuit
    else:
        raise (
            Exception(
                "Ansatz is not parametrizable and no parameters has been provided."
            )
        )
    save_circuit(circuit, "circuit.json")
