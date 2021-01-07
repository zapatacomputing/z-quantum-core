from typing import Union, Optional, List
from numpy.lib.arraysetops import isin
from openfermion import InteractionOperator

from zquantum.core.openfermion import (
    get_fermion_number_operator as _get_fermion_number_operator,
    get_diagonal_component as _get_diagonal_component,
    save_interaction_operator,
    load_interaction_operator,
    load_qubit_operator,
    save_qubit_operator,
)

from zquantum.core.hamiltonian import (
    reorder_fermionic_modes as _reorder_fermionic_modes,
)


def get_fermion_number_operator(
    number_of_qubits: int, number_of_particles: Optional[int] = None
):
    """Get the nubmer operator for the input number of qubits. Optionally, the number of particles can be passed.
    Outputs are serialized to JSON under the file: "number-operator.json"

    ARGS:
        number_of_qubits (int): The number of qubits
        number_of_particles (int): The number of particles
    """
    number_op = _get_fermion_number_operator(number_of_qubits, number_of_particles)
    save_interaction_operator(number_op, "number-operator.json")


def get_diagonal_component(interaction_operator: Union[InteractionOperator, str]):
    """Get the diagonal component and remainder of an input interaction operator. Outputs are serialized to JSON
    under the files: "diagonal-operator.json" and "remainder-operator.json"

    ARGS:
        interaction_operator (Union[InteractionOperator, str]): The input interaction operator
    """
    if isinstance(interaction_operator, str):
        interaction_operator = load_interaction_operator(interaction_operator)

    diagonal_operator, remainder_operator = _get_diagonal_component(
        interaction_operator
    )
    save_interaction_operator(diagonal_operator, "diagonal-operator.json")
    save_interaction_operator(remainder_operator, "remainder-operator.json")


def interpolate_qubit_operators(
    reference_qubit_operator: Union[InteractionOperator, str],
    target_qubit_operator: Union[InteractionOperator, str],
    epsilon: Optional[float] = 0.5,
):
    """Produce a qubit operator which is the interpolation of two operators through the function:
        epsilon * target_qubit_operator + (1.0 - epsilon) * reference_qubit_operator.
    Outputs are serialized to JSON under the file: "qubit-operator.json"

    ARGS:
        reference_qubit_operator (Union[InteractionOperator, str]): The initial operator
        target_qubit_operator (Union[InteractionOperator, str]): The target operator
        epsilon (float): The parameterization between the two operators. Default value is 0.5
    """
    reference_qubit_operator = load_qubit_operator(reference_qubit_operator)
    target_qubit_operator = load_qubit_operator(target_qubit_operator)

    if epsilon > 1.0 or epsilon < 0.0:
        raise ValueError("epsilon must be in the range [0.0, 1.0]")

    output_qubit_operator = (
        epsilon * target_qubit_operator + (1.0 - epsilon) * reference_qubit_operator
    )

    save_qubit_operator(output_qubit_operator, "qubit-operator.json")


def reorder_fermionic_modes(interaction_operator: str, ordering: List) -> InteractionOperator:

    interaction_operator = load_interaction_operator(interaction_operator)

    reordered_operator = _reorder_fermionic_modes(interaction_operator, ordering)
    save_interaction_operator(reordered_operator, "reordered-operator.json")
