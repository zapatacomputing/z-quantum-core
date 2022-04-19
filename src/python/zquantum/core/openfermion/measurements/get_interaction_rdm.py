################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
#   Copyright 2017 The OpenFermion Developers
#   Modifications copyright 2022 Zapata Computing, Inc. for compatibility reasons.
#
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

import itertools

import numpy
from zquantum.core.openfermion.ops.operators import FermionOperator
from zquantum.core.openfermion.ops.representations import InteractionRDM
from zquantum.core.openfermion.transforms.opconversions.conversions import (
    check_no_sympy,
)
from zquantum.core.openfermion.utils.operator_utils import count_qubits


def get_interaction_rdm(qubit_operator, n_qubits=None):
    """Build an InteractionRDM from measured qubit operators.

    Returns: An InteractionRDM object.
    """

    check_no_sympy(qubit_operator)

    # Avoid circular import.
    from zquantum.core.openfermion.transforms import jordan_wigner

    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    one_rdm = numpy.zeros((n_qubits,) * 2, dtype=complex)
    two_rdm = numpy.zeros((n_qubits,) * 4, dtype=complex)

    # One-RDM.
    for i, j in itertools.product(range(n_qubits), repeat=2):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 0))))
        for term, coefficient in transformed_operator.terms.items():
            if term in qubit_operator.terms:
                one_rdm[i, j] += coefficient * qubit_operator.terms[term]

    # Two-RDM.
    for i, j, k, l in itertools.product(range(n_qubits), repeat=4):
        transformed_operator = jordan_wigner(
            FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0)))
        )
        for term, coefficient in transformed_operator.terms.items():
            if term in qubit_operator.terms:
                two_rdm[i, j, k, l] += coefficient * qubit_operator.terms[term]

    return InteractionRDM(one_rdm, two_rdm)
