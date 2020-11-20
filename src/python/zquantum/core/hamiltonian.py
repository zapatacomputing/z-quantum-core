from openfermion.ops import QubitOperator
import numpy as np
from typing import Tuple, List

from .measurement import ExpectationValues

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

def compute_group_variances(groups, expecval=None):
    """Computes the variances of each frame in a grouped operator. 
        If expectation values are provided, use variances from there, 
        otherwise assume variances are 1 (upper bound). Correlation information
        is ignored in the current implementation, covariances are assumed to be 0.
    Args:
        groups:  A list of QubitOperators that defines a (grouped) objective function
        expecval (zquantum.core.measurement.ExpectationValues): 
                  An ExpectationValues object containing the expectation
                  values of the operators and their squares. Optionally, contains
                  values of operator products to compute covariances.
    Returns:
        frame_variances (list): A list of the computed variances for each frame
    """

    if expecval is None:
        frame_variances = [ np.sum(np.array(list(x.terms.values()))**2) for x in groups ] # Covariances are ignored; Variances are set to 1
    else:
        pauli_variances = 1. - np.real(expecval.values)**2
        pauli_variances[pauli_variances < -1.0] = -1.0
        pauli_variances[pauli_variances >  1.0] =  1.0
        group_sizes = np.array([ len(g.terms.keys()) for g in groups ])
        frame_variances = []
        for i, group  in enumerate(groups):
            coeffs = np.array(list(group.terms.values()))
            offset = 0 if i == 0 else np.sum(group_sizes[:i-1])
            frame_variances.append(np.sum(coeffs**2 * pauli_variances[offset : offset + group_sizes[i]]))

    return np.array(frame_variances)

def get_expectation_values_from_rdms(interactionrdm, qubitoperator, sort_terms=False):
    """ Computes expectation values of a qubitOperator
        given a fermionic interactionRDM operator from
        OpenFermion.

        Args:
            interactionrdm (openfermion.ops.InteractionRDM): interaction RDM to use for the expectation values
                computation, as an OF InteractionRDM object
            qubitoperator (openfermion.ops.QubitOperator): qubit operator to compute the expectation values for
                in the form of an OF QubittOperator object
            sort_terms (bool): whether or not the input qubit operator needs to be sorted before calculating expectations
        Returns:
            expectations (zquantum.core.measurement.ExpectationValues): expectation values of Pauli strings in the qubit operator
    """
    if sort_terms:
        terms_iterator = sorted(
            qubit_operator.terms.items(), key=lambda x: abs(x[1]), reverse=True
        )
    else:
        terms_iterator = qubit_operator.terms.items()
    reordered_qubit_operator = QubitOperator((), 0.0)
    for term, coefficient in terms_iterator:
        reordered_qubit_operator += QubitOperator(term, coefficient)

    expectations_packed = interactionrdm.get_qubit_expectations(reordered_qubit_operator)

    expectations = np.real(np.array(expectations_packed.values())) # should we added an assert to catch large Im parts
    # Clip expectations if they fell out of [-1 , 1] due to numerical errors
    expectations[expectations < -1.0] = -1.0
    expectations[expectations >  1.0] =  1.0

    return ExpectationValues(expectations)

