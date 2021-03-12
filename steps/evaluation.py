from pyquil.wavefunction import Wavefunction
import json
import numpy as np
from typing import Dict, Union, List
from openfermion import SymbolicOperator, QubitOperator
from openfermion.linalg import (
    qubit_operator_sparse,
    jw_get_ground_state_at_particle_number as _jw_get_ground_state_at_particle_number,
)

from zquantum.core.measurement import (
    save_expectation_values,
    save_wavefunction,
    load_expectation_values,
    ExpectationValues,
)
from zquantum.core.circuit import (
    load_circuit,
    load_parameter_grid,
    load_circuit_template_params,
    save_circuit_template_params,
    Circuit,
    ParameterGrid,
)
from zquantum.core.utils import (
    create_object,
    ValueEstimate,
    save_value_estimate,
)
from zquantum.core.openfermion import (
    load_qubit_operator,
    load_qubit_operator_set,
    evaluate_operator_for_parameter_grid as _evaluate_operator_for_parameter_grid,
    get_ground_state_rdm_from_qubit_op as _get_ground_state_rdm_from_qubit_op,
    save_interaction_rdm,
    save_parameter_grid_evaluation,
    evaluate_qubit_operator_list as _evaluate_qubit_operator_list,
    convert_dict_to_qubitop,
)
from zquantum.core.typing import Specs


def get_expectation_values_for_qubit_operator(
    backend_specs: Specs,
    circuit: Union[str, Circuit, Dict],
    qubit_operator: Union[str, SymbolicOperator, Dict],
):
    """Measure the expection values of the terms in an input operator with respect to the state prepared by the input
    circuit on the backend described by the backend_specs. The results are serialized into a JSON under the
    file: "expectation-values.json"

    ARGS:
        backend_specs (Union[dict, str]): The backend on which to run the quantum circuit
        circuit (Union[str, Circuit]): The circuit that prepares the state to be measured
        qubit_operator (Union[str, SymbolicOperator]): The operator to measure
    """
    if isinstance(circuit, str):
        circuit = load_circuit(circuit)
    elif isinstance(circuit, dict):
        circuit = Circuit.from_dict(circuit)
    if isinstance(qubit_operator, str):
        qubit_operator = load_qubit_operator(qubit_operator)
    elif isinstance(qubit_operator, dict):
        qubit_operator = convert_dict_to_qubitop(qubit_operator)
    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    backend = create_object(backend_specs)

    expectation_values = backend.get_expectation_values(circuit, qubit_operator)
    save_expectation_values(expectation_values, "expectation-values.json")


def get_ground_state_rdm_from_qubit_operator(
    qubit_operator: Union[str, QubitOperator], n_particles: int
):
    """Diagonalize operator and compute the ground state 1- and 2-RDM

    ARGS:
        qubit_operator (Union[str, QubitOperator]): The openfermion operator to diagonalize
        n_particles (int): number of particles in the target ground state
    """
    qubit_operator = load_qubit_operator(qubit_operator)
    rdm = _get_ground_state_rdm_from_qubit_op(qubit_operator, n_particles)
    save_interaction_rdm(rdm, "rdms.json")


def evaluate_operator_for_parameter_grid(
    ansatz_specs: Specs,
    backend_specs: Specs,
    grid: Union[str, ParameterGrid],
    operator: Union[str, SymbolicOperator],
    fixed_parameters: Union[List[float], np.ndarray, str] = None,
):
    """Measure the expection values of the terms in an input operator with respect to the states prepared by the input
    ansatz circuits when set to the different parameters in the input parameter grid on the
    backend described by the backend_specs. The results are serialized into a JSON under the
    files: "parameter-grid-evaluation.json" and "optimal-parameters.json"

    ARGS:
        ansatz_specs (Union[dict, str]): The ansatz producing the parameterized quantum circuits
        backend_specs (Union[dict, str]): The backend on which to run the quantum circuit
        grid (Union[str, ParameterGrid]): The parameter grid describing the different ansatz parameters to use
        operator (Union[str, SymbolicOperator]): The operator to measure
        fixed_parameters (Union[List[float], np.ndarray, str]): Any fixed parameter values that the ansatz should be
            evaluated to that are not described by the parameter grid
    """
    if isinstance(ansatz_specs, str):
        ansatz_specs = json.loads(ansatz_specs)
    ansatz = create_object(ansatz_specs)

    if isinstance(backend_specs, str):
        backend_specs = json.loads(backend_specs)
    backend = create_object(backend_specs)

    if isinstance(grid, str):
        grid = load_parameter_grid(grid)
    if isinstance(operator, str):
        operator = load_qubit_operator(operator)

    if fixed_parameters is not None:
        if isinstance(fixed_parameters, str):
            fixed_parameters = load_circuit_template_params(fixed_parameters)
    else:
        fixed_parameters = []

    (
        parameter_grid_evaluation,
        optimal_parameters,
    ) = _evaluate_operator_for_parameter_grid(
        ansatz, grid, backend, operator, previous_layer_params=fixed_parameters
    )

    save_parameter_grid_evaluation(
        parameter_grid_evaluation, "parameter-grid-evaluation.json"
    )
    save_circuit_template_params(optimal_parameters, "optimal-parameters.json")


def jw_get_ground_state_at_particle_number(
    particle_number: int, qubit_operator: Union[str, SymbolicOperator]
):
    """Get the ground state wavefunction of the operator for the input particle number. Outputs are serialized to JSON
    within the files: "ground-state.json" and "value-estimate.json"

    ARGS:
        particle_number (int): The given number of particles in the system
        qubit_operator (Union[str, SymbolicOperator]): The operator for which to find the ground state
    """
    if isinstance(qubit_operator, str):
        qubit_operator = load_qubit_operator(qubit_operator)
    sparse_matrix = qubit_operator_sparse(qubit_operator)

    ground_energy, ground_state_amplitudes = _jw_get_ground_state_at_particle_number(
        sparse_matrix, particle_number
    )
    ground_state = Wavefunction(ground_state_amplitudes)
    value_estimate = ValueEstimate(ground_energy)

    save_wavefunction(ground_state, "ground-state.json")
    save_value_estimate(value_estimate, "value-estimate.json")


def evaluate_qubit_operator_list(
    qubit_operator_list: Union[str, List[QubitOperator]],
    expectation_values: Union[str, ExpectationValues],
):
    if isinstance(qubit_operator_list, str):
        qubit_operator_list = load_qubit_operator_set(qubit_operator_list)
    if isinstance(expectation_values, str):
        expectation_values = load_expectation_values(expectation_values)

    value_estimate = _evaluate_qubit_operator_list(
        qubit_operator_list, expectation_values
    )

    save_value_estimate(value_estimate, "value-estimate.json")
