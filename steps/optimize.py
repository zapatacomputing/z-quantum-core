import json
from typing import List, Optional, Union

import numpy as np
import zquantum.core.circuits as new_circuits
from openfermion import SymbolicOperator
from zquantum.core.circuits import Circuit
from zquantum.core.cost_function import (
    AnsatzBasedCostFunction,
    get_ground_state_cost_function,
)
from zquantum.core.estimation import estimate_expectation_values_by_averaging
from zquantum.core.openfermion import load_qubit_operator
from zquantum.core.serialization import (
    load_array,
    save_array,
    save_optimization_results,
)
from zquantum.core.typing import Specs
from zquantum.core.utils import create_object, load_list


def optimize_parametrized_circuit_for_ground_state_of_operator(
    optimizer_specs: Specs,
    target_operator: Union[SymbolicOperator, str],
    parametrized_circuit: Union[Circuit, str],
    backend_specs: Specs,
    estimation_method_specs: Optional[Specs] = None,
    estimation_preprocessors_specs: Optional[List[Specs]] = None,
    initial_parameters: Union[str, np.ndarray, List[float]] = None,
    fixed_parameters: Optional[Union[np.ndarray, str]] = None,
    parameter_precision: Optional[float] = None,
    parameter_precision_seed: Optional[int] = None,
    keep_history: bool = True,
    **kwargs,
):
    """Optimize the parameters of a parametrized quantum circuit to prepare the ground
    state of a target operator.

    Args:
        optimizer_specs: The specs of the optimizer to use to refine the parameter
            values
        target_operator: The operator of which to prepare the ground state
        parametrized_circuit: The parametrized quantum circuit that prepares trial
            states
        backend_specs: The specs of the quantum backend (or simulator) to use to run the
            circuits
        estimation_method_specs: A reference to a callable to use to estimate the
            expectation value of the operator. The default is the
            estimate_expectation_values_by_averaging function.
        estimation_preprocessors_specs: A list of Specs that describe callable functions
            that adhere to the EstimationPreprocessor protocol.
        initial_parameters: The initial parameter values to begin optimization
        fixed_parameters: values for the circuit parameters that should be fixed.
        parameter_precision: the standard deviation of the Gaussian noise to add to each
            parameter, if any.
        parameter_precision_seed: seed for randomly generating parameter deviation if
            using parameter_precision
        keep_history: flag indicating whether to store optimization history.
        kwargs: unused, exists for compatibility
    """
    if isinstance(optimizer_specs, str):
        optimizer_specs = json.loads(optimizer_specs)

    optimizer = create_object(optimizer_specs)

    if isinstance(target_operator, str):
        target_operator = load_qubit_operator(target_operator)

    if isinstance(parametrized_circuit, str):
        with open(parametrized_circuit) as f:
            parametrized_circuit = new_circuits.circuit_from_dict(json.load(f))

    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    backend = create_object(backend_specs)

    if estimation_method_specs is not None:
        if isinstance(estimation_method_specs, str):
            estimation_method_specs = json.loads(estimation_method_specs)
        estimation_method = create_object(estimation_method_specs)
    else:
        estimation_method = estimate_expectation_values_by_averaging

    estimation_preprocessors = []
    if estimation_preprocessors_specs is not None:
        for estimation_preprocessor_specs in estimation_preprocessors_specs:
            if isinstance(estimation_preprocessor_specs, str):
                estimation_preprocessor_specs = json.loads(
                    estimation_preprocessor_specs
                )
            estimation_preprocessors.append(
                create_object(estimation_preprocessor_specs)
            )

    if initial_parameters is not None:
        if isinstance(initial_parameters, str):
            initial_parameters = load_array(initial_parameters)

    if fixed_parameters is not None:
        if isinstance(fixed_parameters, str):
            fixed_parameters = load_array(fixed_parameters)

    cost_function = get_ground_state_cost_function(
        target_operator,
        parametrized_circuit,
        backend,
        estimation_method=estimation_method,
        estimation_preprocessors=estimation_preprocessors,
        fixed_parameters=fixed_parameters,
        parameter_precision=parameter_precision,
        parameter_precision_seed=parameter_precision_seed,
    )

    optimization_results = optimizer.minimize(
        cost_function, initial_parameters, keep_history
    )

    save_optimization_results(optimization_results, "optimization-results.json")
    save_array(optimization_results.opt_params, "optimized-parameters.json")


def optimize_ansatz_based_cost_function(
    optimizer_specs: Specs,
    target_operator: Union[SymbolicOperator, str],
    ansatz_specs: Specs,
    backend_specs: Specs,
    estimation_method_specs: Optional[Specs] = None,
    estimation_preprocessors_specs: Optional[List[Specs]] = None,
    initial_parameters: Union[str, np.ndarray, List[float]] = None,
    fixed_parameters: Optional[Union[np.ndarray, str]] = None,
    parameter_precision: Optional[float] = None,
    parameter_precision_seed: Optional[int] = None,
    keep_history: bool = False,
    **kwargs,
):
    """Optimize the parameters of an ansatz circuit to prepare the ground state of a
    target operator.

    Args:
        optimizer_specs: The specs of the optimizer to use to refine the parameter
            values
        target_operator: The operator of which to prepare the ground state
        ansatz_specs: The specs describing an Ansatz which will prepare the quantum
            circuit
        backend_specs: The specs of the quantum backend (or simulator) to use to run the
            circuits
        estimation_method_specs: A reference to a callable to use to estimate the
            expectation value of the operator. The default is the
            estimate_expectation_values_by_averaging function.
        estimation_preprocessors_specs: A list of Specs that describe callable functions
            that adhere to the EstimationPreprocessor protocol.
        initial_parameters: The initial parameter values to begin optimization
        fixed_parameters: values for the circuit parameters that should be fixed.
        parameter_precision: the standard deviation of the Gaussian noise to add to each
            parameter, if any.
        parameter_precision_seed: seed for randomly generating parameter deviation if
            using parameter_precision
        keep_history: flag indicating whether to store optimization history.
        kwargs:
            The following key word arguments are handled explicitly when appropriate:
                - thetas: A list of thetas used to initialize the WarmStartQAOAAnsatz
    """
    if isinstance(optimizer_specs, str):
        optimizer_specs = json.loads(optimizer_specs)

    optimizer = create_object(optimizer_specs)

    if isinstance(target_operator, str):
        target_operator = load_qubit_operator(target_operator)

    if isinstance(ansatz_specs, str):
        ansatz_specs = json.loads(ansatz_specs)

    if "WarmStartQAOAAnsatz" in ansatz_specs["function_name"]:
        ansatz_specs["thetas"] = np.array(load_list(kwargs.pop("thetas")))
        ansatz_specs["cost_hamiltonian"] = target_operator
    elif "QAOA" in ansatz_specs["function_name"]:
        ansatz_specs["cost_hamiltonian"] = target_operator
    ansatz = create_object(ansatz_specs)

    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    backend = create_object(backend_specs)

    if estimation_method_specs is not None:
        if isinstance(estimation_method_specs, str):
            estimation_method_specs = json.loads(estimation_method_specs)
        estimation_method = create_object(estimation_method_specs)
    else:
        estimation_method = estimate_expectation_values_by_averaging

    estimation_preprocessors = []
    if estimation_preprocessors_specs is not None:
        for estimation_preprocessor_specs in estimation_preprocessors_specs:
            if isinstance(estimation_preprocessor_specs, str):
                estimation_preprocessor_specs = json.loads(
                    estimation_preprocessor_specs
                )
            estimation_preprocessors.append(
                create_object(estimation_preprocessor_specs)
            )

    if initial_parameters is not None:
        if isinstance(initial_parameters, str):
            initial_parameters = load_array(initial_parameters)

    if fixed_parameters is not None:
        if isinstance(fixed_parameters, str):
            fixed_parameters = load_array(fixed_parameters)

    cost_function = AnsatzBasedCostFunction(
        target_operator,
        ansatz,
        backend,
        estimation_method=estimation_method,
        estimation_preprocessors=estimation_preprocessors,
        fixed_parameters=fixed_parameters,
        parameter_precision=parameter_precision,
        parameter_precision_seed=parameter_precision_seed,
    )

    optimization_results = optimizer.minimize(
        cost_function, initial_parameters, keep_history
    )

    save_optimization_results(optimization_results, "optimization-results.json")
    save_array(optimization_results.opt_params, "optimized-parameters.json")
