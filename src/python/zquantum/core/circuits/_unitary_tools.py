"""Module containing utilities for handling unitary matrices."""
from functools import reduce

import numpy as np
import sympy


def _permute(vector, permutation):
    """Permute vector according to given ordering of indices.

    Args:
        vector: 1-D array-like object
        permutation: any permutation of indices 0...len(vector)
    Returns:
        permuted vector
    Examples:
        _permute([4, 5, 6], [0, 2, 1]) == [4, 6, 5]
    """
    return [vector[i] for i in permutation]


def _basis_bitstring(i, num_qubits):
    """Create vector corresponding to i-th basis vector of num_qubits system."""
    return [int(char) for char in bin(i)[2:].zfill(num_qubits)]


def _bitstring_to_sympy_dense_vector(state):
    """Construct sympy matrix representation of a state given by a bitstring."""
    basis = [sympy.Matrix([1, 0]), sympy.Matrix([0, 1])]
    return sympy.kronecker_product(*[basis[bit] for bit in state])


def _bitstring_to_numpy_dense_vector(state):
    """Construct numpy array representation of a state given by a bitstring."""
    basis = [np.array([1, 0]), np.array([0, 1])]
    return reduce(np.kron, (basis[bit] for bit in state))


def _permutation_matrix(target_indices_order, zeros, bitstring_to_dense_vector):
    """Construct a permutation matrix for N qubit system.

    Args:
        target_indices_order: the desired order of systems (qubits) after permutation.
           This should contain all entries from 0 to N-1, where N is the number of
           qubits.
        zeros: function for constructing all-zero matrix that will be filled with
            appropriate entries.
        bitstring_to_dense_vector: function constructing dense vector from bitstring.
    Returns:
        Matrix permuting qubits according to given ordering.
    Notes:
        Matrix acts on whole N-qubit system, regardless of how many qubits actually
        change their position. The `zeros` and `bitstring_to_dense_vector` functions are
        used for constructing variants of this function depending on whether sympy or
        numpy should be used (see below how this dispatching is done).
    """
    num_qubits = len(target_indices_order)
    if sorted(target_indices_order) != list(range(num_qubits)):
        raise ValueError("Not all qubits given in permutation.")
    perm_matrix = zeros((2 ** num_qubits, 2 ** num_qubits))
    for i in range(2 ** num_qubits):
        input_state = _basis_bitstring(i, num_qubits)
        output_state = _permute(input_state, target_indices_order)
        perm_matrix[:, i] = bitstring_to_dense_vector(output_state)
    return perm_matrix


def _permutation_making_qubits_adjacent(qubit_indices, num_qubits):
    """Given an iterable of qubit indices construct a permutation such that they are
    next to each other."""
    return list(qubit_indices) + [
        i for i in range(num_qubits) if i not in qubit_indices
    ]


def _lift_matrix(
    matrix,
    qubit_indices,
    num_qubits,
    zeros,
    eye,
    kronecker_product,
    bitstring_to_dense_vector,
):
    """Lift a matrix acting on subsystem of N-qubit system to one acting on the whole
    system.

    Args:
        matrix: matrix acting on k `qubits`
        qubit_indices: indices of qubits that matrix acts on
        zeros: function constructing all-zero matrix. It is assumed that it can be
            called like `zeros((m, n))`.
        eye: function constructing identity matrix
        kronecker_product: function computing kronecker product of matrices.
            It is assumed that it can be applied like `kronecker_product(x, y)`.
        bitstring_to_dense_vector: function converting bitstring to dense vector.
    Returns:
        Matrix that acts like `matrix` on qubits with indices `qubit_indices` and as
            identity on other qubits.
    """
    smallest, largest = min(qubit_indices), max(qubit_indices)
    # No need to consider all the qubits, just those between smallest and largest one
    shifted_qubits = [index - smallest for index in qubit_indices]
    inner_permutation = _permutation_making_qubits_adjacent(
        shifted_qubits, largest - smallest + 1
    )
    # perm_matrix permutes qubits in range smallest-largest so that the active
    # ones come first.
    perm_matrix = _permutation_matrix(
        inner_permutation, zeros, bitstring_to_dense_vector
    )

    # inner_gate_matrix acts on the whole range smallest-largest, by
    # transforming first qubits in the same way as matrix, and leaving
    # others unchanged
    inner_gate_matrix = kronecker_product(
        matrix, eye(2 ** (largest - smallest - len(qubit_indices) + 1))
    )
    # target operation acting on smallest-largest range is now composed of
    # 1. basis change (via permutation)
    # 2. gate action
    # 3. reversal of basis change
    inner_matrix = perm_matrix.transpose() @ inner_gate_matrix @ perm_matrix
    # finally, to make a matrix acting on whole range of qubits, we
    # add identities acting on qubits with indices outside of smallest-largest range.
    return reduce(
        kronecker_product,
        [eye(2 ** smallest), inner_matrix, eye(2 ** (num_qubits - largest - 1))],
    )


def _lift_matrix_numpy(matrix, qubits, num_qubits):
    """A version of _lift_matrix working on numpy arrays.

    Notice that the input matrix is typically sympy's matrix, so we have to first
    convert it.
    """
    matrix = np.array(matrix, dtype=complex)
    return _lift_matrix(
        matrix,
        qubits,
        num_qubits,
        np.zeros,
        np.eye,
        np.kron,
        _bitstring_to_numpy_dense_vector,
    )


def _lift_matrix_sympy(matrix, qubits, num_qubits):
    """A version of _lift_matrix working on Sympy matrices."""
    return _lift_matrix(
        matrix,
        qubits,
        num_qubits,
        # sympy has different signature of `zeros` than numpy, hence the below adapter.
        lambda size: sympy.zeros(*size),
        sympy.eye,
        sympy.kronecker_product,
        _bitstring_to_sympy_dense_vector,
    )
