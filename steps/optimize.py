from zquantum.core.cost_function import get_ground_state_cost_function
from zquantum.core.estimator import BasicEstimator
from zquantum.core.circuit import load_circuit, load_circuit_template_params, Circuit
from zquantum.core.utils import create_object
from zquantum.core.serialization import save_optimization_results
from openfermion import SymbolicOperator
import json
import os
import numpy as np
from typing import Union, Dict, Optional, List


def optimize_parameterized_quantum_circuit(
    optimizer_specs: Union[Dict, str],
    target_operator: Union[SymbolicOperator, str],
    circuit: Union[Circuit, str],
    backend_specs: Union[Dict, str],
    estimator_specs: Union[Dict, str] = None,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    initial_parameters: Union[str, np.ndarray, List[float]] = None,
):
    # for input_argument in [estimator_specs, epsilon, delta, initial_parameters]:
    if estimator_specs == "None":
        estimator_specs = None
    if epsilon == "None":
        epsilon = None
    if delta == "None":
        delta = None
    if initial_parameters == "None":
        initial_parameters = None

    if isinstance(optimizer_specs, str):
        optimizer_specs = json.loads(optimizer_specs)
    optimizer = create_object(optimizer_specs)

    if isinstance(target_operator, str):
        with open(target_operator, "r") as f:
            target_operator = json.loads(f.read())

    if isinstance(circuit, str):
        circuit = load_circuit(circuit)

    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    backend = create_object(backend_specs)

    if estimator_specs is not None:
        if isinstance(estimator_specs, str):
            estimator_specs = json.loads(estimator_specs)
        estimator = create_object(estimator_specs)
    else:
        estimator = BasicEstimator()

    if initial_parameters is not None:
        if isinstance(initial_parameters, str):
            initial_parameters = load_circuit_template_params(initial_parameters)

    cost_function = get_ground_state_cost_function(
        target_operator,
        circuit,
        backend,
        estimator=estimator,
        epsilon=epsilon,
        delta=delta,
    )

    optimization_results = optimizer.minimize(cost_function, initial_parameters)

    save_optimization_results(optimization_results, "optimization_results.json")
