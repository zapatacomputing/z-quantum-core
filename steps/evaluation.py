import json
from typing import Dict, List, Union

import zquantum.core.wip.circuits as new_circuits
from openfermion import QubitOperator, SymbolicOperator
from openfermion.linalg import (
    jw_get_ground_state_at_particle_number as _jw_get_ground_state_at_particle_number,
)
from openfermion.linalg import qubit_operator_sparse
from pyquil.wavefunction import Wavefunction
from zquantum.core.measurement import (
    ExpectationValues,
    load_expectation_values,
    save_expectation_values,
    save_wavefunction,
)
from zquantum.core.openfermion import convert_dict_to_qubitop
from zquantum.core.openfermion import (
    evaluate_qubit_operator_list as _evaluate_qubit_operator_list,
)
from zquantum.core.openfermion import (
    get_ground_state_rdm_from_qubit_op as _get_ground_state_rdm_from_qubit_op,
)
from zquantum.core.openfermion import (
    load_qubit_operator,
    load_qubit_operator_set,
    save_interaction_rdm,
)
from zquantum.core.typing import Specs
from zquantum.core.utils import ValueEstimate, create_object, save_value_estimate


def get_expectation_values_for_qubit_operator(
    backend_specs: Specs,
    circuit: Union[str, new_circuits.Circuit, Dict],
    qubit_operator: Union[str, SymbolicOperator, Dict],
):
    """Measure the expectation values of the terms in an input operator with respect to
    the state prepared by the input circuit on the backend described by the
    `backend_specs`. The results are serialized into a JSON under the file:
    "expectation-values.json"

    Args:
        backend_specs: The backend on which to run the quantum circuit
        circuit: The circuit that prepares the state to be measured
        qubit_operator: The operator to measure
    """
    if isinstance(circuit, str):
        circuit = new_circuits.load_circuit(circuit)
    elif isinstance(circuit, dict):
        circuit = new_circuits.circuit_from_dict(circuit)
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

    Args:
        qubit_operator: The openfermion operator to diagonalize
        n_particles: number of particles in the target ground state
    """
    qubit_operator = load_qubit_operator(qubit_operator)
    rdm = _get_ground_state_rdm_from_qubit_op(qubit_operator, n_particles)
    save_interaction_rdm(rdm, "rdms.json")


def jw_get_ground_state_at_particle_number(
    particle_number: int, qubit_operator: Union[str, SymbolicOperator]
):
    """Get the ground state wavefunction of the operator for the input particle number.

    Outputs are serialized to JSON within the files: "ground-state.json" and
    "value-estimate.json".

    Args:
        particle_number: The given number of particles in the system
        qubit_operator: The operator for which to find the ground state
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
