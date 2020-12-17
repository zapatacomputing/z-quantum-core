from openfermion.ops import QubitOperator, InteractionRDM, InteractionOperator
import numpy as np
from typing import Tuple, List, Optional

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


def compute_group_variances(
    groups: List[QubitOperator], expecval: ExpectationValues = None
) -> np.array:
    """Computes the variances of each frame in a grouped operator. 

    If expectation values are provided, use variances from there, 
    otherwise assume variances are 1 (upper bound). Correlation information
    is ignored in the current implementation, covariances are assumed to be 0.

    Args:
        groups:  A list of QubitOperators that defines a (grouped) objective function
        expecval: An ExpectationValues object containing the expectation
            values of the operators and their squares. Optionally, contains
            values of operator products to compute covariances.
    Returns:
        frame_variances: A Numpy array of the computed variances for each frame
    """

    if expecval is None:
        frame_variances = [
            np.sum(np.array(list(x.terms.values())) ** 2) for x in groups
        ]  # Covariances are ignored; Variances are set to 1
    else:
        group_sizes = np.array([len(g.terms.keys()) for g in groups])
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


def estimate_nmeas(
    target_operator: QubitOperator,
    decomposition_method: Optional[str] = "greedy-sorted",
    expecval: Optional[ExpectationValues] = None,
) -> Tuple[float, int, np.array]:
    """Calculates the number of measurements required for computing
        the expectation value of a qubit hamiltonian, where co-measurable terms
        are grouped. 

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
        target_operator: A QubitOperator to measure
        expecval: An ExpectationValues object containing the expectation
            values of the operators and their squares. Optionally, contains
            values of operator products to compute covariances.
            If absent, covariances are assumed to be 0 and variances are
            assumed to be maximal, i.e. 1. It is assumed that the first expectation
            value corresponds to the constant term in the target operator in line 
            with the conventions used in the BasicEstimator
            NOTE: WE HAVE TO MAKE SURE THAT THE ORDER
            OF EXPECTATION VALUES MATCHES THE ORDER OF THE TERMS IN THE
            TARGET QUBIT OPERATOR, OTHERWISE THIS FUNCTION WILL NOT WORK CORRECTLY
    Returns:
        K2: number of measurements for epsilon = 1.0
        nterms: number of groups of QWC terms in the target_operator
        frame_meas: Number of optimal measurements per group 
    """

    frame_variances = None
    if decomposition_method == "greedy-sorted":
        decomposition_function = lambda qubit_operator: group_comeasureable_terms_greedy(
            qubit_operator, True
        )
    elif decomposition_method == "greedy":
        decomposition_function = lambda qubit_operator: group_comeasureable_terms_greedy(
            qubit_operator, False
        )
    else:
        raise ValueError(f"{decomposition_method} grouping is not implemented")

    groups = decomposition_function(target_operator)
    frame_variances = compute_group_variances(groups, expecval)
    sqrt_lambda = sum(np.sqrt(frame_variances))
    frame_meas = sqrt_lambda * np.sqrt(frame_variances)
    K2 = sum(frame_meas)
    nterms = sum([len(group.terms) for group in groups])

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
