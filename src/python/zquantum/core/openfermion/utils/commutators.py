#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Module to compute commutators, with optimizations for specific systems."""

import numpy
from zquantum.core.openfermion.ops.operators import FermionOperator
from zquantum.core.openfermion.transforms.opconversions.term_reordering import (
    normal_ordered,
)


def commutator(operator_a, operator_b):
    """Compute the commutator of two operators.

    Args:
        operator_a, operator_b: Operators in commutator. Any operators
            are accepted so long as implicit subtraction and multiplication are
            supported; e.g. QubitOperators, FermionOperators, BosonOperators,
            or Scipy sparse matrices. 2D Numpy arrays are also supported.

    Raises:
        TypeError: operator_a and operator_b are not of the same type.
    """
    if type(operator_a) != type(operator_b):
        raise TypeError("operator_a and operator_b are not of the same type.")
    if isinstance(operator_a, numpy.ndarray):
        result = operator_a.dot(operator_b)
        result -= operator_b.dot(operator_a)
    else:
        result = operator_a * operator_b
        result -= operator_b * operator_a
    return result


def anticommutator(operator_a, operator_b):
    """Compute the anticommutator of two operators.

    Args:
        operator_a, operator_b: Operators in anticommutator. Any operators
            are accepted so long as implicit addition and multiplication are
            supported; e.g. QubitOperators, FermionOperators, BosonOperators,
            or Scipy sparse matrices. 2D Numpy arrays are also supported.

    Raises:
        TypeError: operator_a and operator_b are not of the same type.
    """
    if type(operator_a) != type(operator_b):
        raise TypeError("operator_a and operator_b are not of the same type.")
    if isinstance(operator_a, numpy.ndarray):
        result = operator_a.dot(operator_b)
        result += operator_b.dot(operator_a)
    else:
        result = operator_a * operator_b
        result += operator_b * operator_a
    return result


def double_commutator(
    op1,
    op2,
    op3,
    indices2=None,
    indices3=None,
    is_hopping_operator2=None,
    is_hopping_operator3=None,
):
    """Return the double commutator [op1, [op2, op3]].

    Args:
        op1, op2, op3 (FermionOperators or BosonOperators): operators for
            the commutator. All three operators must be of the same type.
        indices2, indices3 (set): The indices op2 and op3 act on.
        is_hopping_operator2 (bool): Whether op2 is a hopping operator.
        is_hopping_operator3 (bool): Whether op3 is a hopping operator.

    Returns:
        The double commutator of the given operators.
    """
    if is_hopping_operator2 and is_hopping_operator3:
        indices2 = set(indices2)
        indices3 = set(indices3)
        # Determine which indices both op2 and op3 act on.
        try:
            (intersection,) = indices2.intersection(indices3)
        except ValueError:
            return FermionOperator.zero()

        # Remove the intersection from the set of indices, since it will get
        # cancelled out in the final result.
        indices2.remove(intersection)
        indices3.remove(intersection)

        # Find the indices of the final output hopping operator.
        (index2,) = indices2
        (index3,) = indices3
        coeff2 = op2.terms[list(op2.terms)[0]]
        coeff3 = op3.terms[list(op3.terms)[0]]
        commutator23 = FermionOperator(
            ((index2, 1), (index3, 0)), coeff2 * coeff3
        ) + FermionOperator(((index3, 1), (index2, 0)), -coeff2 * coeff3)
    else:
        commutator23 = normal_ordered(commutator(op2, op3))

    return normal_ordered(commutator(op1, commutator23))
