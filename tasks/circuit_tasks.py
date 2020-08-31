import numpy as np
from zquantum.core.circuit import save_circuit_template_params
from zquantum.core.utils import create_object

def generate_random_ansatz_params(ansatz_specs, seed=None, min_value, max_value):

    if ansatz_specs is not None:
        ansatz = create_object(ansatz_specs)
        number_of_params = ansatz.number_of_params
    if seed is not None:
        np.random.seed(seed)
    params = np.random.uniform(min_value, max_value, number_of_params)
    save_circuit_template_params(params, 'params.json')