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
"""This module provides functions to interface with scipy.sparse."""
import itertools
from functools import reduce

import numpy
import numpy.linalg
import scipy
import scipy.sparse
import scipy.sparse.linalg
from zquantum.core.openfermion.ops.operators import FermionOperator, QubitOperator
from zquantum.core.openfermion.ops.representations import PolynomialTensor
from zquantum.core.openfermion.transforms.opconversions import normal_ordered
from zquantum.core.openfermion.utils.operator_utils import count_qubits, is_hermitian

# Make global definitions.
identity_csc = scipy.sparse.identity(2, format="csc", dtype=complex)
pauli_x_csc = scipy.sparse.csc_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
pauli_y_csc = scipy.sparse.csc_matrix([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
pauli_z_csc = scipy.sparse.csc_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
q_raise_csc = (pauli_x_csc - 1.0j * pauli_y_csc) / 2.0
q_lower_csc = (pauli_x_csc + 1.0j * pauli_y_csc) / 2.0
pauli_matrix_map = {
    "I": identity_csc,
    "X": pauli_x_csc,
    "Y": pauli_y_csc,
    "Z": pauli_z_csc,
}


def wrapped_kronecker(operator_1, operator_2):
    """Return the Kronecker product of two sparse.csc_matrix operators."""
    return scipy.sparse.kron(operator_1, operator_2, "csc")


def kronecker_operators(*args):
    """Return the Kronecker product of multiple sparse.csc_matrix operators."""
    return reduce(wrapped_kronecker, *args)


def jordan_wigner_ladder_sparse(n_qubits, tensor_factor, ladder_type):
    r"""Make a matrix representation of a fermion ladder operator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Args:
        index: This is a nonzero integer. The integer indicates the tensor
            factor and the sign indicates raising or lowering.
        n_qubits(int): Number qubits in the system Hilbert space.

    Returns:
        The corresponding Scipy sparse matrix.
    """
    parities = tensor_factor * [pauli_z_csc]
    identities = [
        scipy.sparse.identity(
            2 ** (n_qubits - tensor_factor - 1), dtype=complex, format="csc"
        )
    ]
    if ladder_type:
        operator = kronecker_operators(parities + [q_raise_csc] + identities)
    else:
        operator = kronecker_operators(parities + [q_lower_csc] + identities)
    return operator


def jordan_wigner_sparse(fermion_operator, n_qubits=None):
    r"""Initialize a Scipy sparse matrix from a FermionOperator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Args:
        fermion_operator(FermionOperator): instance of the FermionOperator
            class.
        n_qubits(int): Number of qubits.

    Returns:
        The corresponding Scipy sparse matrix.
    """
    if n_qubits is None:
        n_qubits = count_qubits(fermion_operator)

    # Create a list of raising and lowering operators for each orbital.
    jw_operators = []
    for tensor_factor in range(n_qubits):
        jw_operators += [
            (
                jordan_wigner_ladder_sparse(n_qubits, tensor_factor, 0),
                jordan_wigner_ladder_sparse(n_qubits, tensor_factor, 1),
            )
        ]

    # Construct the Scipy sparse matrix.
    n_hilbert = 2 ** n_qubits
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]
        sparse_matrix = coefficient * scipy.sparse.identity(
            2 ** n_qubits, dtype=complex, format="csc"
        )
        for ladder_operator in term:
            sparse_matrix = (
                sparse_matrix * jw_operators[ladder_operator[0]][ladder_operator[1]]
            )

        if coefficient:
            # Extract triplets from sparse_term.
            sparse_matrix = sparse_matrix.tocoo(copy=False)
            values_list.append(sparse_matrix.data)
            (row, column) = sparse_matrix.nonzero()
            row_list.append(row)
            column_list.append(column)

    values_list = numpy.concatenate(values_list)
    row_list = numpy.concatenate(row_list)
    column_list = numpy.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix(
        (values_list, (row_list, column_list)), shape=(n_hilbert, n_hilbert)
    ).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def qubit_operator_sparse(qubit_operator, n_qubits=None):
    """Initialize a Scipy sparse matrix from a QubitOperator.

    Args:
        qubit_operator(QubitOperator): instance of the QubitOperator class.
        n_qubits (int): Number of qubits.

    Returns:
        The corresponding Scipy sparse matrix.
    """
    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    if n_qubits < count_qubits(qubit_operator):
        raise ValueError("Invalid number of qubits specified.")

    # Construct the Scipy sparse matrix.
    n_hilbert = 2 ** n_qubits
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]

    # Loop through the terms.
    for qubit_term in qubit_operator.terms:
        tensor_factor = 0
        coefficient = qubit_operator.terms[qubit_term]
        sparse_operators = [coefficient]
        for pauli_operator in qubit_term:

            # Grow space for missing identity operators.
            if pauli_operator[0] > tensor_factor:
                identity_qubits = pauli_operator[0] - tensor_factor
                identity = scipy.sparse.identity(
                    2 ** identity_qubits, dtype=complex, format="csc"
                )
                sparse_operators += [identity]

            # Add actual operator to the list.
            sparse_operators += [pauli_matrix_map[pauli_operator[1]]]
            tensor_factor = pauli_operator[0] + 1

        # Grow space at end of string unless operator acted on final qubit.
        if tensor_factor < n_qubits or not qubit_term:
            identity_qubits = n_qubits - tensor_factor
            identity = scipy.sparse.identity(
                2 ** identity_qubits, dtype=complex, format="csc"
            )
            sparse_operators += [identity]

        # Extract triplets from sparse_term.
        sparse_matrix = kronecker_operators(sparse_operators)
        values_list.append(sparse_matrix.tocoo(copy=False).data)
        (column, row) = sparse_matrix.nonzero()
        column_list.append(column)
        row_list.append(row)

    # Create sparse operator.
    values_list = numpy.concatenate(values_list)
    row_list = numpy.concatenate(row_list)
    column_list = numpy.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix(
        (values_list, (row_list, column_list)), shape=(n_hilbert, n_hilbert)
    ).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def jw_configuration_state(occupied_orbitals, n_qubits):
    """Function to produce a basis state in the occupation number basis.

    Args:
        occupied_orbitals(list): A list of integers representing the indices
            of the occupied orbitals in the desired basis state
        n_qubits(int): The total number of qubits

    Returns:
        basis_vector(sparse): The basis state as a sparse matrix
    """
    one_index = sum(2 ** (n_qubits - 1 - i) for i in occupied_orbitals)
    basis_vector = numpy.zeros(2 ** n_qubits, dtype=float)
    basis_vector[one_index] = 1
    return basis_vector


def jw_hartree_fock_state(n_electrons, n_orbitals):
    """Function to produce Hartree-Fock state in JW representation."""
    hartree_fock_state = jw_configuration_state(range(n_electrons), n_orbitals)
    return hartree_fock_state


def jw_number_indices(n_electrons, n_qubits):
    """Return the indices for n_electrons in n_qubits under JW encoding

    Calculates the indices for all possible arrangements of n-electrons
        within n-qubit orbitals when a Jordan-Wigner encoding is used.
        Useful for restricting generic operators or vectors to a particular
        particle number space when desired

    Args:
        n_electrons(int): Number of particles to restrict the operator to
        n_qubits(int): Number of qubits defining the total state

    Returns:
        indices(list): List of indices in a 2^n length array that indicate
            the indices of constant particle number within n_qubits
            in a Jordan-Wigner encoding.
    """
    occupations = itertools.combinations(range(n_qubits), n_electrons)
    indices = [sum([2 ** n for n in occupation]) for occupation in occupations]
    return indices


def jw_number_restrict_operator(operator, n_electrons, n_qubits=None):
    """Restrict a Jordan-Wigner encoded operator to a given particle number

    Args:
        sparse_operator(ndarray or sparse): Numpy operator acting on
            the space of n_qubits.
        n_electrons(int): Number of particles to restrict the operator to
        n_qubits(int): Number of qubits defining the total state

    Returns:
        new_operator(ndarray or sparse): Numpy operator restricted to
            acting on states with the same particle number.
    """
    if n_qubits is None:
        n_qubits = int(numpy.log2(operator.shape[0]))

    select_indices = jw_number_indices(n_electrons, n_qubits)
    return operator[numpy.ix_(select_indices, select_indices)]


def jw_get_ground_state_at_particle_number(sparse_operator, particle_number):
    """Compute ground energy and state at a specified particle number.

    Assumes the Jordan-Wigner transform. The input operator should be Hermitian
    and particle-number-conserving.

    Args:
        sparse_operator(sparse): A Jordan-Wigner encoded sparse matrix.
        particle_number(int): The particle number at which to compute the ground
            energy and states

    Returns:
        ground_energy(float): The lowest eigenvalue of sparse_operator within
            the eigenspace of the number operator corresponding to
            particle_number.
        ground_state(ndarray): The ground state at the particle number
    """

    n_qubits = int(numpy.log2(sparse_operator.shape[0]))

    # Get the operator restricted to the subspace of the desired particle number
    restricted_operator = jw_number_restrict_operator(
        sparse_operator, particle_number, n_qubits
    )

    # Compute eigenvalues and eigenvectors
    if restricted_operator.shape[0] - 1 <= 1:
        # Restricted operator too small for sparse eigensolver
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = numpy.linalg.eigh(dense_restricted_operator)
    else:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(
            restricted_operator, k=1, which="SA"
        )

    # Expand the state
    state = eigvecs[:, 0]
    expanded_state = numpy.zeros(2 ** n_qubits, dtype=complex)
    expanded_state[jw_number_indices(particle_number, n_qubits)] = state

    return eigvals[0], expanded_state


def get_density_matrix(states, probabilities):
    n_qubits = states[0].shape[0]
    density_matrix = scipy.sparse.csc_matrix((n_qubits, n_qubits), dtype=complex)
    for state, probability in zip(states, probabilities):
        state = scipy.sparse.csc_matrix(state.reshape((len(state), 1)))
        density_matrix = density_matrix + probability * state * state.getH()
    return density_matrix


def get_ground_state(sparse_operator, initial_guess=None):
    """Compute lowest eigenvalue and eigenstate.

    Args:
        sparse_operator (LinearOperator): Operator to find the ground state of.
        initial_guess (ndarray): Initial guess for ground state.  A good
            guess dramatically reduces the cost required to converge.

    Returns
    -------
        eigenvalue:
            The lowest eigenvalue, a float.
        eigenstate:
            The lowest eigenstate in scipy.sparse csc format.
    """
    values, vectors = scipy.sparse.linalg.eigsh(
        sparse_operator, k=1, v0=initial_guess, which="SA", maxiter=1e7
    )

    order = numpy.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    eigenvalue = values[0]
    eigenstate = vectors[:, 0]
    return eigenvalue, eigenstate.T


def eigenspectrum(operator, n_qubits=None):
    """Compute the eigenspectrum of an operator.

    WARNING: This function has cubic runtime in dimension of
        Hilbert space operator, which might be exponential.

    NOTE: This function does not currently support
        QuadOperator and BosonOperator.

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            PolynomialTensor, or InteractionRDM.
        n_qubits (int): number of qubits/modes in operator. if None, will
            be counted.

    Returns:
        spectrum: dense numpy array of floats giving eigenspectrum.
    """
    sparse_operator = get_sparse_operator(operator, n_qubits)
    spectrum = sparse_eigenspectrum(sparse_operator)
    return spectrum


def sparse_eigenspectrum(sparse_operator):
    """Perform a dense diagonalization.

    Returns:
        eigenspectrum: The lowest eigenvalues in a numpy array.
    """
    dense_operator = sparse_operator.todense()
    if is_hermitian(sparse_operator):
        eigenspectrum = numpy.linalg.eigvalsh(dense_operator)
    else:
        eigenspectrum = numpy.linalg.eigvals(dense_operator)
    return numpy.sort(eigenspectrum)


def expectation(operator, state):
    """Compute the expectation value of an operator with a state.

    Args:
        operator(scipy.sparse.spmatrix or scipy.sparse.linalg.LinearOperator):
            The operator whose expectation value is desired.
        state(numpy.ndarray or scipy.sparse.spmatrix): A numpy array
            representing a pure state or a sparse matrix representing a density
            matrix. If `operator` is a LinearOperator, then this must be a
            numpy array.

    Returns:
        A complex number giving the expectation value.

    Raises:
        ValueError: Input state has invalid format.
    """

    if isinstance(state, scipy.sparse.spmatrix):
        # Handle density matrix.
        if isinstance(operator, scipy.sparse.linalg.LinearOperator):
            raise ValueError(
                "Taking the expectation of a LinearOperator with "
                "a density matrix is not supported."
            )
        product = state * operator
        expectation = numpy.sum(product.diagonal())

    elif isinstance(state, numpy.ndarray):
        # Handle state vector.
        if len(state.shape) == 1:
            # Row vector
            expectation = numpy.dot(numpy.conjugate(state), operator * state)
        else:
            # Column vector
            expectation = numpy.dot(numpy.conjugate(state.T), operator * state)[0, 0]

    else:
        # Handle exception.
        raise ValueError("Input state must be a numpy array or a sparse matrix.")

    # Return.
    return expectation


def inner_product(state_1, state_2):
    """Compute inner product of two states."""
    return numpy.dot(state_1.conjugate(), state_2)


def get_sparse_operator(operator, n_qubits=None, trunc=None, hbar=1.0):
    r"""Map an operator to a sparse matrix.

    If the input is not a QubitOperator, the Jordan-Wigner Transform is used.

    Args:
        operator: Currently supported operators include:
            FermionOperator, QubitOperator, PolynomialTensor.
        n_qubits(int): Number qubits in the system Hilbert space.
            Applicable only to fermionic systems.
        trunc (int): The size at which the Fock space should be truncated.
            Applicable only to bosonic systems.
        hbar (float): the value of hbar to use in the definition of the
            canonical commutation relation [q_i, p_j] = \delta_{ij} i hbar.
            Applicable only to the QuadOperator.
    """
    from zquantum.core.openfermion.transforms.opconversions import get_fermion_operator

    if isinstance(operator, PolynomialTensor):
        return jordan_wigner_sparse(get_fermion_operator(operator))
    elif isinstance(operator, FermionOperator):
        return jordan_wigner_sparse(operator, n_qubits)
    elif isinstance(operator, QubitOperator):
        return qubit_operator_sparse(operator, n_qubits)
    else:
        raise TypeError(
            "Failed to convert a {} to a sparse matrix.".format(type(operator).__name__)
        )


def get_number_preserving_sparse_operator(
    fermion_op,
    num_qubits,
    num_electrons,
    spin_preserving=False,
    reference_determinant=None,
    excitation_level=None,
):
    """Initialize a Scipy sparse matrix in a specific symmetry sector.

    This method initializes a Scipy sparse matrix from a FermionOperator,
    explicitly working in a particular particle number sector. Optionally, it
    can also restrict the space to contain only states with a particular Sz.

    Finally, the Hilbert space can also be restricted to only those states
    which are reachable by excitations up to a fixed rank from an initial
    reference determinant.

    Args:
        fermion_op(FermionOperator): An instance of the FermionOperator class.
            It should not contain terms which do not preserve particle number.
            If spin_preserving is set to True it should also not contain terms
            which do not preserve the Sz (it is assumed that the ordering of
            the indices goes alpha, beta, alpha, beta, ...).
        num_qubits(int): The total number of qubits / spin-orbitals in the
            system.
        num_electrons(int): The number of particles in the desired Hilbert
            space.
        spin_preserving(bool): Whether or not the constructed operator should
            be defined in a space which has support only on states with the
            same Sz value as the reference_determinant.
        reference_determinant(list(bool)): A list, whose length is equal to
            num_qubits, which specifies which orbitals should be occupied in
            the reference state. If spin_preserving is set to True then the Sz
            value of this reference state determines the Sz value of the
            symmetry sector in which the generated operator acts. If a value
            for excitation_level is provided then the excitations are generated
            with respect to the reference state. In any case, the ordering of
            the states in the matrix representation of the operator depends on
            reference_determinant and the state corresponding to
            reference_determinant is the vector [1.0, 0.0, 0.0 ... 0.0]. Can be
            set to None in order to take the first num_electrons orbitals to be
            the occupied orbitals.
        excitation_level(int): The number of excitations from the reference
            state which should be included in the generated operator's matrix
            representation. Can be set to None to include all levels of
            excitation.

    Returns:
        sparse_op(scipy.sparse.csc_matrix): A sparse matrix representation of
            fermion_op in the basis set by the arguments.
    """

    # We use the Hartree-Fock determinant as a reference if none is provided.
    if reference_determinant is None:
        reference_determinant = numpy.array(
            [i < num_electrons for i in range(num_qubits)]
        )
    else:
        reference_determinant = numpy.asarray(reference_determinant)

    if excitation_level is None:
        excitation_level = num_electrons

    state_array = numpy.asarray(
        list(_iterate_basis_(reference_determinant, excitation_level, spin_preserving))
    )
    # Create a 1d array with each determinant encoded
    # as an integer for sorting purposes.
    int_state_array = state_array.dot(1 << numpy.arange(state_array.shape[1])[::-1])
    sorting_indices = numpy.argsort(int_state_array)

    space_size = state_array.shape[0]

    fermion_op = normal_ordered(fermion_op)

    sparse_op = scipy.sparse.csc_matrix((space_size, space_size), dtype=float)

    for term, coefficient in fermion_op.terms.items():
        if len(term) == 0:
            constant = coefficient * scipy.sparse.identity(
                space_size, dtype=float, format="csc"
            )

            sparse_op += constant

        else:
            term_op = _build_term_op_(
                term, state_array, int_state_array, sorting_indices
            )

            sparse_op += coefficient * term_op

    return sparse_op


def _iterate_basis_(reference_determinant, excitation_level, spin_preserving):
    """A helper method which iterates over the specified basis states.

    Note that this method always yields the states in order of their excitation
    rank from the reference_determinant.

    Args:
        reference_determinant(list(bool)): A list of bools which indicates
            which orbitals are occupied and which are unoccupied in the
            reference state.
        excitation_level(int): The maximum excitation rank to iterate over.
        spin_preserving(bool): A bool which, if set to True, constrains the
            method to iterate over only those states which have the same Sz as
            reference_determinant.

    Yields:
        Lists of bools which indicate which orbitals are occupied and which are
            unoccupied in the current determinant.
    """
    if not spin_preserving:
        for order in range(excitation_level + 1):
            for determinant in _iterate_basis_order_(reference_determinant, order):
                yield determinant

    else:
        alpha_excitation_level = min(
            (numpy.sum(reference_determinant[::2]), excitation_level)
        )
        beta_excitation_level = min(
            (numpy.sum(reference_determinant[1::2]), excitation_level)
        )

        for order in range(excitation_level + 1):
            for alpha_order in range(alpha_excitation_level + 1):
                beta_order = order - alpha_order
                if beta_order < 0 or beta_order > beta_excitation_level:
                    continue

                for determinant in _iterate_basis_spin_order_(
                    reference_determinant, alpha_order, beta_order
                ):
                    yield determinant


def _iterate_basis_order_(reference_determinant, order):
    """A helper for iterating over determinants of a fixed excitation rank.

    Args:
        reference_determinant(list(bool)): The reference state with respect to
            which we are iterating over excited determinants.
        order(int): The number of excitations from the modes which are occupied
            in the reference_determinant.

    Yields:
        Lists of bools which indicate which orbitals are occupied and which are
            unoccupied in the current determinant.
    """
    occupied_indices = numpy.where(reference_determinant)[0]
    unoccupied_indices = numpy.where(numpy.invert(reference_determinant))[0]

    for occ_ind, unocc_ind in itertools.product(
        itertools.combinations(occupied_indices, order),
        itertools.combinations(unoccupied_indices, order),
    ):
        basis_state = reference_determinant.copy()

        occ_ind = list(occ_ind)
        unocc_ind = list(unocc_ind)

        basis_state[occ_ind] = False
        basis_state[unocc_ind] = True

        yield basis_state


def _iterate_basis_spin_order_(reference_determinant, alpha_order, beta_order):
    """Iterates over states with a fixed excitation rank for each spin sector.

    This helper method assumes that the two spin sectors are interleaved:
    [1_alpha, 1_beta, 2_alpha, 2_beta, ...].

    Args:
        reference_determinant(list(bool)): The reference state with respect to
            which we are iterating over excited determinants.
        alpha_order(int): The number of excitations from the alpha spin sector
            of the reference_determinant.
        beta_order(int): The number of excitations from the beta spin sector of
            the reference_determinant.

    Yields:
        Lists of bools which indicate which orbitals are occupied and which are
            unoccupied in the current determinant.
    """
    occupied_alpha_indices = numpy.where(reference_determinant[::2])[0] * 2
    unoccupied_alpha_indices = (
        numpy.where(numpy.invert(reference_determinant[::2]))[0] * 2
    )
    occupied_beta_indices = numpy.where(reference_determinant[1::2])[0] * 2 + 1
    unoccupied_beta_indices = (
        numpy.where(numpy.invert(reference_determinant[1::2]))[0] * 2 + 1
    )

    for (
        alpha_occ_ind,
        alpha_unocc_ind,
        beta_occ_ind,
        beta_unocc_ind,
    ) in itertools.product(
        itertools.combinations(occupied_alpha_indices, alpha_order),
        itertools.combinations(unoccupied_alpha_indices, alpha_order),
        itertools.combinations(occupied_beta_indices, beta_order),
        itertools.combinations(unoccupied_beta_indices, beta_order),
    ):
        basis_state = reference_determinant.copy()

        alpha_occ_ind = list(alpha_occ_ind)
        alpha_unocc_ind = list(alpha_unocc_ind)
        beta_occ_ind = list(beta_occ_ind)
        beta_unocc_ind = list(beta_unocc_ind)

        basis_state[alpha_occ_ind] = False
        basis_state[alpha_unocc_ind] = True
        basis_state[beta_occ_ind] = False
        basis_state[beta_unocc_ind] = True

        yield basis_state


def _build_term_op_(term, state_array, int_state_array, sorting_indices):
    """Builds a scipy sparse representation of a term from a FermionOperator.

    Args:
        term(tuple of tuple(int, int)s): The argument is a tuple of tuples
            representing a product of normal ordered fermionic creation and
            annihilation operators, each of which is of the form (int, int)
            where the first int indicates which site the operator acts on and
            the second int indicates whether the operator is a creation
            operator (1) or an annihilation operator (0). See the
            implementation of FermionOperator for more details.
        state_array(ndarray(bool)): A Numpy array which encodes each of the
            determinants in the space we are working in a bools which indicate
            the occupation of each mode. See the implementation of
            get_number_preserving_sparse_operator for more details.
        int_state_array(ndarray(int)): A one dimensional Numpy array which
            encodes the intFeger representation of the binary number
            corresponding to each determinant in state_array.
        sorting_indices(ndarray.view): A Numpy view which sorts
            int_state_array. This, together with int_state_array, allows for a
            quick lookup of the position of a particular determinant in
            state_array by converting it to its integer representation and
            searching through the sorted int_state_array.

    Raises:
        ValueError: If term does not represent a particle number conserving
            operator.

    Returns:
        A scipy.sparse.csc_matrix which corresponds to the operator specified
            by term expressed in the basis corresponding to the other arguments
            of the method."""

    space_size = state_array.shape[0]

    needs_to_be_occupied = []
    needs_to_be_unoccupied = []

    # We keep track of the number of creation and annihilation operators and
    # ensure that there are an equal number of them in order to help detect
    # invalid inputs.
    delta = 0
    for index, op_type in reversed(term):
        if op_type == 0:
            needs_to_be_occupied.append(index)
            delta -= 1
        else:
            if index not in needs_to_be_occupied:
                needs_to_be_unoccupied.append(index)
            delta += 1

    if delta != 0:
        raise ValueError("The supplied operator doesn't preserve particle number")

    # We search for every state which has the necessary orbitals occupied and
    # unoccupied in order to not be immediately zeroed out based on the
    # creation and annihilation operators specified in term.
    maybe_valid_states = numpy.where(
        numpy.logical_and(
            numpy.all(state_array[:, needs_to_be_occupied], axis=1),
            numpy.logical_not(
                numpy.any(state_array[:, needs_to_be_unoccupied], axis=1)
            ),
        )
    )[0]

    data = []
    row_ind = []
    col_ind = []
    shape = (space_size, space_size)

    # For each state that is not immediately zeroed out by the action of our
    # operator we check to see if the determinant which this state gets mapped
    # to is in the space we are considering.
    # Note that a failure to find any state does not necessarily indicate that
    # term specifies an invalid operator. For example, if we are restricting
    # ourselves to double excitations from a fixed reference state then the
    # action of term on some of our basis states may lead to determinants with
    # more than two excitations from the reference. These more than double
    # excited determinants are not included in the matrix representation (and
    # hence, will not be present in state_array).
    for _, state in enumerate(maybe_valid_states):
        determinant = state_array[state, :]
        target_determinant = determinant.copy()

        parity = 1
        for i, _ in reversed(term):
            area_to_check = target_determinant[0:i]
            parity *= (-1) ** numpy.sum(area_to_check)

            target_determinant[i] = not target_determinant[i]

        int_encoding = target_determinant.dot(
            1 << numpy.arange(target_determinant.size)[::-1]
        )

        target_state_index_sorted = numpy.searchsorted(
            int_state_array, int_encoding, sorter=sorting_indices
        )

        target_state = sorting_indices[target_state_index_sorted]

        if int_state_array[target_state] == int_encoding:
            # Then target state is in the space considered:
            data.append(parity)
            row_ind.append(target_state)
            col_ind.append(state)

    data = numpy.asarray(data)
    row_ind = numpy.asarray(row_ind)
    col_ind = numpy.asarray(col_ind)

    term_op = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)

    return term_op
