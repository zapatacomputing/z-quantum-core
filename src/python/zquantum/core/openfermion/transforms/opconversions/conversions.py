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

import sympy
from zquantum.core.openfermion.ops.operators import FermionOperator
from zquantum.core.openfermion.ops.representations import PolynomialTensor
from zquantum.core.openfermion.utils.operator_utils import count_qubits


def check_no_sympy(operator):
    """Checks whether a SymbolicOperator contains any
    sympy expressions, which will prevent it being converted
    to a PolynomialTensor

    Args:
        operator(SymbolicOperator): the operator to be tested
    """
    for key in operator.terms:
        if isinstance(operator.terms[key], sympy.Expr):
            raise TypeError(
                "This conversion is currently not supported "
                + "for operators with sympy expressions "
                + "as coefficients"
            )


def get_fermion_operator(operator):
    """Convert to FermionOperator.

    Returns:
        fermion_operator: An instance of the FermionOperator class.
    """
    if isinstance(operator, PolynomialTensor):
        return _polynomial_tensor_to_fermion_operator(operator)
    else:
        raise TypeError(
            "{} cannot be converted to FermionOperator".format(type(operator))
        )


def _polynomial_tensor_to_fermion_operator(operator):
    fermion_operator = FermionOperator()
    for term in operator:
        fermion_operator += FermionOperator(term, operator[term])
    return fermion_operator
