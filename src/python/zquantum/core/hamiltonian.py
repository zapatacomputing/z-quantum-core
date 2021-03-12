from openfermion.ops import QubitOperator, InteractionRDM, InteractionOperator
import numpy as np
from typing import Tuple, List, Optional, Callable

from .measurement import ExpectationValues, expectation_values_to_real


def is_comeasureable(
    term_1: Tuple[Tuple[int, str], ...], term_2: Tuple[Tuple[int, str], ...]
) -> bool:
    """Determine if two Pauli terms are co-measureable. Co-measureable means that
       for each qubit: if one term contains a Pauli operator acting on a qubit,
       then the other term cannot have a different Pauli operator acting on that
       qubit.
    Args:
        term1: a product of Pauli operators represented in openfermion style
        term2: a product of Pauli operators represented in openfermion style
    Returns:
        bool: True if the terms are co-measureable.
    """

    for qubit_1, operator_1 in term_1:
        for qubit_2, operator_2 in term_2:

            # Check if the two Pauli operators act on the same qubit
            if qubit_1 == qubit_2:

                # Check if the two Pauli operators are different
                if operator_1 != operator_2:
                    return False

    return True


def group_comeasureable_terms_greedy(
    qubit_operator: QubitOperator, sort_terms: bool = False
) -> List[QubitOperator]:
    """Group co-measurable terms in a qubit operator using a greedy algorithm. Adapted from pyquil.
    Args:
        qubit_operator: the operator whose terms are to be grouped
        sort_terms: whether to sort terms by the absolute value of the coefficient when grouping
        Returns:
        A list of qubit operators.
    """

    groups = []  # List of QubitOperators representing groups of co-measureable terms

    if sort_terms:
        terms_iterator = sorted(
            qubit_operator.terms.items(), key=lambda x: abs(x[1]), reverse=True
        )
    else:
        terms_iterator = qubit_operator.terms.items()

    for term, coefficient in terms_iterator:
        assigned = False  # True if the current term has been assigned to a group
        # Identity should not be in a group
        if term == ():
            continue
        for group in groups:
            if all(
                is_comeasureable(term, term_to_compare)
                for term_to_compare in group.terms
            ):
                # Add the term to the group
                group += QubitOperator(term, coefficient)
                assigned = True
                break

        # If term was not co-measureable with any group, it gets to start its own group!
        if not assigned:
            groups.append(QubitOperator(term, qubit_operator.terms[term]))

    return groups


DECOMPOSITION_METHODS = {
    "greedy": group_comeasureable_terms_greedy,
    "greedy-sorted": lambda qubit_operator: group_comeasureable_terms_greedy(
        qubit_operator, True
    ),
}


def get_decomposition_function(
    decomposition_method: str,
) -> Callable[[QubitOperator], List[QubitOperator]]:
    """Get a function for Hamiltonian decomposition from its name.

    Args:
        decomposition_method: The name of the Hamiltonian decomposition method.

    Returns:
        A callable that performs the decomposition.
    """

    decomposition_function = DECOMPOSITION_METHODS.get(decomposition_method)
    if decomposition_function is None:
        raise ValueError(
            f"Unrecognized decomposition method {decomposition_method}. Allowed values are {list(DECOMPOSITION_METHODS.keys())}"
        )
    return decomposition_function


def compute_group_variances(
    groups: List[QubitOperator], expecval: ExpectationValues = None
) -> np.array:
    """Computes the variances of each frame in a grouped operator.

    If expectation values are provided, use variances from there,
    otherwise assume variances are 1 (upper bound). Correlation information
    is ignored in the current implementation, covariances are assumed to be 0.

    Args:
        groups:  A list of QubitOperators that defines a (grouped) operator
        expecval: An ExpectationValues object containing the expectation
            values of the operators.
    Returns:
        frame_variances: A Numpy array of the computed variances for each frame
    """

    if any([group.terms.get(()) for group in groups]):
        raise ValueError(
            "The list of qubitoperators for measurement estimation should not contain a constant term"
        )
    if expecval is None:
        frame_variances = [
            np.sum(np.array(list(group.terms.values())) ** 2) for group in groups
        ]  # Covariances are ignored; Variances are set to 1
    else:
        group_sizes = np.array([len(group.terms.keys()) for group in groups])
        assert np.sum(group_sizes) == len(expecval.values)
        real_expecval = expectation_values_to_real(expecval)
        pauli_variances = 1.0 - real_expecval.values ** 2
        frame_variances = []
        for i, group in enumerate(groups):
            coeffs = np.array(list(group.terms.values()))
            offset = 0 if i == 0 else np.sum(group_sizes[:i])
            pauli_variances_for_group = pauli_variances[
                offset : offset + group_sizes[i]
            ]
            frame_variances.append(np.sum(coeffs ** 2 * pauli_variances_for_group))

    return np.array(frame_variances)


def get_expectation_values_from_rdms_for_qubitoperator_list(
    interactionrdm: InteractionRDM,
    qubitoperator_list: List[QubitOperator],
    sort_terms: bool = False,
) -> ExpectationValues:
    """Computes expectation values of Pauli strings in a list of QubitOperator given a
       fermionic InteractionRDM from OpenFermion. All the expectation values for the
       operators in the list are returned in a single ExpectationValues object in the
       same order the operators came in.

    Args:
        interactionrdm (InteractionRDM): interaction RDM to use for the expectation values
            computation, as an OpenFermion InteractionRDM object
        qubitoperator_list (List[QubitOperator]): List of qubit operators to compute the expectation values for
            in the form of OpenFermion QubitOperator objects
        sort_terms (bool): whether or not each input qubit operator needs to be sorted before calculating expectations
    Returns:
        expectation values of Pauli strings in all qubit operators as an ExpectationValues object
    """

    all_expectations = []
    for qubitoperator in qubitoperator_list:
        expectations = get_expectation_values_from_rdms(
            interactionrdm, qubitoperator, sort_terms=sort_terms
        )
        all_expectations += list(expectations.values)

    return ExpectationValues(np.asarray(all_expectations))


def get_expectation_values_from_rdms(
    interactionrdm: InteractionRDM,
    qubitoperator: QubitOperator,
    sort_terms: bool = False,
) -> ExpectationValues:
    """Computes expectation values of Pauli strings in a QubitOperator given a fermionic InteractionRDM from
       OpenFermion.

    Args:
        interactionrdm: interaction RDM to use for the expectation values
            computation, as an OF InteractionRDM object
        qubitoperator: qubit operator to compute the expectation values for
            in the form of an OpenFermion QubitOperator object
        sort_terms: whether or not the input qubit operator needs to be sorted before calculating expectations
    Returns:
        expectation values of Pauli strings in the qubit operator as an ExpectationValues object
    """
    if sort_terms:
        terms_iterator = sorted(
            qubitoperator.terms.items(), key=lambda x: abs(x[1]), reverse=True
        )
    else:
        terms_iterator = qubitoperator.terms.items()
    reordered_qubitoperator = QubitOperator()
    for term, coefficient in terms_iterator:
        reordered_qubitoperator += QubitOperator(term, coefficient)

    expectations_packed = interactionrdm.get_qubit_expectations(reordered_qubitoperator)

    if () in expectations_packed.terms:
        del expectations_packed.terms[
            ()
        ]  # Expectation of the constant term is excluded from expectation values

    expectations = np.array(list(expectations_packed.terms.values()))
    if np.any(np.abs(np.imag(expectations)) > 1e-3):
        raise RuntimeWarning(
            f"Expectation values extracted from rdms inside get_expectation_values_from_rdms are complex!"
        )
    expectations = np.real(expectations)
    np.clip(expectations, -1, 1, out=expectations)

    return ExpectationValues(expectations)


def estimate_nmeas_for_operator(
    operator: QubitOperator,
    decomposition_method: Optional[str] = "greedy-sorted",
    expecval: Optional[ExpectationValues] = None,
):
    """Calculates the number of measurements required for computing
    the expectation value of a qubit hamiltonian, where co-measurable terms
    are grouped. See estimate_nmeas_for_frames for details.

    Args:
        operator: The operator whose expectation value is to be estimated.
        decomposition_method: Method used to decompose the Hamiltonian into
            co-measurable groups. See DECOMPOSITION_METHODS.
        expval: Expectation values to be used when accounting for variances. See
            estimate_nmeas_for_frames for details.

    Returns:
        See estimate_nmeas_for_frames.
    """

    decomposition_function = get_decomposition_function(decomposition_method)
    return estimate_nmeas_for_frames(decomposition_function(operator), expecval)


def estimate_nmeas_for_frames(
    frame_operators: List[QubitOperator],
    expecval: Optional[ExpectationValues] = None,
) -> Tuple[float, int, np.array]:
    """Calculates the number of measurements required for computing
    the expectation value of a qubit hamiltonian, where co-measurable terms
    are grouped in a single QubitOperator, and different groups are different
    members of the list.

    We are assuming the exact expectation values are provided
    (i.e. infinite number of measurements or simulations without noise)
    M ~ (\sum_{i} prec(H_i)) ** 2.0 / (epsilon ** 2.0)
    where prec(H_i) is the precision (square root of the variance)
    for each group of co-measurable terms H_{i}. It is computed as
    prec(H_{i}) = \sum{ab} |h_{a}^{i}||h_{b}^{i}| cov(O_{a}^{i}, O_{b}^{i})
    where h_{a}^{i} is the coefficient of the a-th operator, O_{a}^{i}, in the
    i-th group. Covariances are assumed to be zero for a != b:
    cov(O_{a}^{i}, O_{b}^{i}) = <O_{a}^{i} O_{b}^{i}> - <O_{a}^{i}> <O_{b}^{i}> = 0

    Args:
        frame_operators (List[QubitOperator]): A list of QubitOperator objects, where each element in
            the list is a group of co-measurable terms.
        expecval (Optional[ExpectationValues]): An ExpectationValues object containing the expectation
            values of all operators in frame_operators. If absent, variances are assumed to be
            maximal, i.e. 1.
            NOTE: YOU HAVE TO MAKE SURE THAT THE ORDER OF EXPECTATION VALUES MATCHES
            THE ORDER OF THE TERMS IN THE *GROUPED* TARGET QUBIT OPERATOR, OTHERWISE
            THIS FUNCTION WILL NOT RETURN THE CORRECT RESULT.

    Returns:
        K2 (float): number of measurements for epsilon = 1.0
        nterms (int): number of groups in frame_operators
        frame_meas (np.array): Number of optimal measurements per group
    """
    frame_variances = compute_group_variances(frame_operators, expecval)
    sqrt_lambda = sum(np.sqrt(frame_variances))
    frame_meas = sqrt_lambda * np.sqrt(frame_variances)
    K2 = sum(frame_meas)
    nterms = sum([len(group.terms) for group in frame_operators])

    return K2, nterms, frame_meas


def reorder_fermionic_modes(
    interaction_op: InteractionOperator, ordering: List
) -> InteractionOperator:
    """Reorder the fermionic modes according to a specified ordering.

    Args:
        interaction_op: The input interaction operator.
        ordering: List containing the mode indexes from the input
            operator. For example, an ordering of [0, 2, 1, 3] will
            map mode 2 of the input operator to mode 1.

    Returns:
        An interaction operator with the modes reordered to have the desired ordering.
    """

    one_body_tensor = interaction_op.one_body_tensor[:, :]
    one_body_tensor = one_body_tensor[ordering, :]
    one_body_tensor = one_body_tensor[:, ordering]

    two_body_tensor = interaction_op.two_body_tensor[:, :, :, :]
    two_body_tensor = two_body_tensor[ordering, :, :, :]
    two_body_tensor = two_body_tensor[:, ordering, :, :]
    two_body_tensor = two_body_tensor[:, :, ordering, :]
    two_body_tensor = two_body_tensor[:, :, :, ordering]

    reordered_interaction_op = InteractionOperator(
        interaction_op.constant, one_body_tensor, two_body_tensor
    )

    return reordered_interaction_op
