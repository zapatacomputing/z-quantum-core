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

import numpy
from scipy.sparse import dok_matrix
from zquantum.core.openfermion.ops.operators import FermionOperator
from zquantum.core.openfermion.utils import count_qubits

from .rdm_equality_constraints import two_body_fermion_constraints


def linearize_term(term, n_orbitals):
    """Function to return integer index of term indices.

    Args:
        term(tuple): The term indices of a one- or two-body FermionOperator.
        n_orbitals(int): The number of orbitals in the simulation.

    Returns:
        index(int): The index of the term.
    """
    # Handle identity term.
    if term == ():
        return 0
    elif len(term) == 2:
        # Handle one-body terms.
        assert term[0][1] == 1
        assert term[1][1] == 0
        p = term[0][0]
        q = term[1][0]
        return 1 + p + q * n_orbitals
    elif len(term) == 4:
        # Handle two-body terms.
        assert term[0][1] == 1
        assert term[1][1] == 1
        assert term[2][1] == 0
        assert term[3][1] == 0
        p = term[0][0]
        q = term[1][0]
        r = term[2][0]
        s = term[3][0]
        return (
            1
            + n_orbitals**2
            + p
            + q * n_orbitals
            + r * n_orbitals**2
            + s * n_orbitals**3
        )


def unlinearize_term(index, n_orbitals):
    """Function to return integer index of term indices.

    Args:
        index(int): The index of the term.
        n_orbitals(int): The number of orbitals in the simulation.

    Returns:
        term(tuple): The term indices of a one- or two-body FermionOperator.
    """
    # Handle identity term.
    if not index:
        return ()
    elif 0 < index < 1 + n_orbitals**2:
        # Handle one-body terms.
        shift = 1
        new_index = index - shift
        q = new_index // n_orbitals
        p = new_index - q * n_orbitals
        assert index == shift + p + q * n_orbitals
        return ((p, 1), (q, 0))
    else:
        # Handle two-body terms.
        shift = 1 + n_orbitals**2
        new_index = index - shift
        s = new_index // n_orbitals**3
        r = (new_index - s * n_orbitals**3) // n_orbitals**2
        q = (new_index - s * n_orbitals**3 - r * n_orbitals**2) // n_orbitals
        p = new_index - q * n_orbitals - r * n_orbitals**2 - s * n_orbitals**3
        assert index == (
            shift + p + q * n_orbitals + r * n_orbitals**2 + s * n_orbitals**3
        )
        return ((p, 1), (q, 1), (r, 0), (s, 0))


def constraint_matrix(n_orbitals, n_fermions):
    """Function to generate matrix of constraints.

    Args:
        n_orbitals(int): The number of orbitals in the simulation.
        n_fermions(int): The number of particles in the simulation.

    Returns:
        constraint_matrix(scipy.sparse.coo_matrix): The matrix of constraints.
    """
    # Very inefficiently count constraints.
    n_constraints = 0
    for constraint in two_body_fermion_constraints(n_orbitals, n_fermions):
        n_constraints += 1

    # Initialize constraint matrix.
    n_terms = 1 + n_orbitals**2 + n_orbitals**4
    constraint_matrix = dok_matrix((n_constraints, n_terms))

    # Populate constraint matrix.
    constraint_number = 0
    for constraint in two_body_fermion_constraints(n_orbitals, n_fermions):
        for term, coefficient in constraint.terms.items():
            term_index = linearize_term(term, n_orbitals)
            constraint_matrix[constraint_number, term_index] = coefficient
        constraint_number += 1
    return constraint_matrix


def operator_to_vector(operator):
    """Function to map operator to vector.

    Args:
        operator(FermionOperator): FermionOperator with only 1- and 2-body
            terms that we wish to vectorize.

    Returns:
        vectorized_operator(numpy.array): Vector of term coefficients.
    """
    n_orbitals = count_qubits(operator)
    n_terms = 1 + n_orbitals**2 + n_orbitals**4
    vectorized_operator = numpy.zeros(n_terms, float)
    for term, coefficient in operator.terms.items():
        term_index = linearize_term(term, n_orbitals)
        vectorized_operator[term_index] = coefficient
    return vectorized_operator


def vector_to_operator(vector, n_orbitals):
    """Function to map vector to operator.

    Args:
        vectorized_operator(numpy.array): Vector of term coefficients.

    Returns:
        operator(FermionOperator): FermionOperator with only 1- and 2-body
            terms that we wish to vectorize.
    """
    operator = FermionOperator()
    for index, coefficient in enumerate(vector):
        term = unlinearize_term(index, n_orbitals)
        operator += FermionOperator(term, coefficient)
    return operator
