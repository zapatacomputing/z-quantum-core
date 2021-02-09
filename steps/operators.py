from typing import Union, Optional, List
from numpy.lib.arraysetops import isin
from openfermion import InteractionOperator, FermionOperator, QubitOperator
from openfermion import normal_ordered
from openfermion.transforms import get_fermion_operator
import numpy as np

from zquantum.core.openfermion import (
    get_fermion_number_operator as _get_fermion_number_operator,
    get_diagonal_component as _get_diagonal_component,
    save_interaction_operator,
    load_interaction_operator,
    load_qubit_operator,
    save_qubit_operator,
    remove_inactive_orbitals as _remove_inactive_orbitals,
    save_qubit_operator_set,
    load_qubit_operator_set,
)

from zquantum.core.hamiltonian import (
    reorder_fermionic_modes as _reorder_fermionic_modes,
    group_comeasureable_terms_greedy as _group_comeasurable_terms_greedy,
)

from zquantum.core.testing import create_random_qubitop as _create_random_qubitop


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


def reorder_fermionic_modes(
    interaction_operator: str, ordering: List
) -> InteractionOperator:

    interaction_operator = load_interaction_operator(interaction_operator)

    reordered_operator = _reorder_fermionic_modes(interaction_operator, ordering)
    save_interaction_operator(reordered_operator, "reordered-operator.json")

def get_one_qubit_hydrogen_hamiltonian(interaction_operator: Union[InteractionOperator, str]):
    """Generate a one qubit H2 hamiltonian from a corresponding interaction operator. 

    Original H2 hamiltonian will be reduced to a 2 x 2 matrix defined on a subspace spanned
    by |0011> and |1100> and expanded in terms of I, X, Y, and Z matrices

    ARGS:
        interaction_operator (Union[InteractionOperator, str]): The input interaction operator
    """
    if isinstance(interaction_operator, str):
        interaction_operator = load_interaction_operator(interaction_operator)

    fermion_h = get_fermion_operator(interaction_operator)

    # H00
    H00 = normal_ordered(FermionOperator('0 1') * fermion_h * FermionOperator('1^ 0^'))
    H00 = H00.terms[()]

    # H11
    H11 = normal_ordered(FermionOperator('2 3') * fermion_h * FermionOperator('3^ 2^'))
    H11 = H11.terms[()]

    # H10
    H10 = normal_ordered(FermionOperator('2 3') * fermion_h * FermionOperator('1^ 0^'))
    H10 = H10.terms[()]

    # H01
    H01 = np.conj(H10)

    one_qubit_h_matrix = np.array([[H00, H01], [H10, H11]])
    pauli_x = np.array([[0., 1.], [1., 0.]])
    pauli_y = np.array([[0., -1.j], [1.j, 0.]])
    pauli_z = np.array([[1., 0.], [0., -1.]])

    r_id = 0.5 * np.trace(one_qubit_h_matrix)
    r_x = 0.5 * np.trace(one_qubit_h_matrix @ pauli_x)
    r_y = 0.5 * np.trace(one_qubit_h_matrix @ pauli_y)
    r_z = 0.5 * np.trace(one_qubit_h_matrix @ pauli_z)

    one_qubit_h = r_id * QubitOperator('') + r_x * QubitOperator('X0') + r_y * QubitOperator('Y0') + r_z * QubitOperator('Z0')

    save_qubit_operator(one_qubit_h, 'qubit-operator.json')

def remove_inactive_orbitals(
    interaction_operator: str,
    n_active: Optional[int] = None,
    n_core: Optional[int] = None,
):

    interaction_operator = load_interaction_operator(interaction_operator)

    frozen_operator = _remove_inactive_orbitals(
        interaction_operator, n_active=n_active, n_core=n_core
    )

    save_interaction_operator(frozen_operator, "frozen-operator.json")


def create_one_qubit_operator(
    x_coeff: float, y_coeff: float, z_coeff: float, constant: float
) -> None:

    qubit_operator = (
        x_coeff * QubitOperator("X0")
        + y_coeff * QubitOperator("Y0")
        + z_coeff * QubitOperator("Z0")
        + constant * QubitOperator(())
    )
    save_qubit_operator(qubit_operator, "qubit_operator.json")


def group_comeasureable_terms_greedy(
    qubit_operator: Union[str, QubitOperator], sort_terms: bool = False
):

    if isinstance(qubit_operator, str):
        qubit_operator = load_qubit_operator(qubit_operator)

    groups = _group_comeasurable_terms_greedy(qubit_operator, sort_terms=sort_terms)

    save_qubit_operator_set(groups, "grouped-operator.json")


def concatenate_qubit_operator_lists(
    qubit_operator_list_A: Union[str, List[QubitOperator]],
    qubit_operator_list_B: Union[str, List[QubitOperator]],
):
    if isinstance(qubit_operator_list_A, str):
        qubit_operator_list_A = load_qubit_operator_set(qubit_operator_list_A)
    if isinstance(qubit_operator_list_B, str):
        qubit_operator_list_B = load_qubit_operator_set(qubit_operator_list_B)

    qubit_operator_list_final = qubit_operator_list_A + qubit_operator_list_B

    save_qubit_operator_set(
        qubit_operator_list_final, "concatenated-qubit-operators.json"
    )


def create_random_qubitop(nqubits: int, nterms: int):

    output_qubit_operator = _create_random_qubitop(nqubits, nterms)

    save_qubit_operator(output_qubit_operator, "qubit-operator.json")
