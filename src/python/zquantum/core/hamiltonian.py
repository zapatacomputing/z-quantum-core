from openfermion.ops import QubitOperator
from typing import Tuple, List


def is_comeasureable(
    term1: Tuple[Tuple[int, str], ...], term2: Tuple[Tuple[int, str], ...]
) -> bool:
    """Determine if two Pauli terms are co-measureable. Co-measureable means that
       for each qubit: both terms apply the same Pauli operator, or at least one term
        applies the identity.
    Args:
        term1: a product of Pauli operators represented in openfermion style
        term2: a product of Pauli operators represented in openfermion style
    Returns:
        bool: True if the terms are co-measureable.
    """

    for op1 in term1:
        for op2 in term2:

            # Check if the two Pauli operators act on the same qubit
            if op1[0] == op2[0]:

                # Check if the two Pauli operators are different
                if op1[1] != op2[1]:
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
            commeasureable_with_group = True  # True if the current term is co-measureable with the current group
            for term_to_compare in group.terms:
                if not is_comeasureable(term, term_to_compare):
                    commeasureable_with_group = False
                    break
            if commeasureable_with_group:
                # Add the term to the group
                group += QubitOperator(term, coefficient)
                assigned = True
                break

        # If term was not co-measureable with any group, it gets to start its own group!
        if not assigned:
            groups.append(QubitOperator(term, qubit_operator.terms[term]))

    return groups
