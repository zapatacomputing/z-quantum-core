import numpy as np
from json import loads
from zquantum.core.circuit import save_circuit_template_params
from zquantum.core.utils import create_object

def generate_random_ansatz_params(ansatz_specs, min_value, max_value, seed=None):
    if ansatz_specs is not None:
        ansatz_specs_dict = loads(ansatz_specs)
        ansatz = create_object(ansatz_specs)
        number_of_params = ansatz.number_of_params
    if seed is not None:
        np.random.seed(seed)
    params = np.random.uniform(min_value, max_value, number_of_params)
    save_circuit_template_params(params, 'params.json')