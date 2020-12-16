from openfermion import (
    FermionOperator,
    QubitOperator,
    count_qubits,
    InteractionOperator,
    PolynomialTensor,
)
from openfermion.utils import expectation as openfermion_expectation
from openfermion.utils import number_operator, normal_ordered
from openfermion.transforms import get_sparse_operator, get_interaction_operator
import numpy as np
import random
import copy
from typing import List, Union, Optional

from zquantum.core.circuit import build_ansatz_circuit
from zquantum.core.utils import bin2dec, dec2bin, ValueEstimate
from zquantum.core.measurement import ExpectationValues, expectation_values_to_real
from openfermion import count_qubits
import itertools
import cirq


def get_qubitop_from_matrix(operator: List[List]) -> QubitOperator:
    r"""Expands a 2^n by 2^n matrix into n-qubit Pauli basis. The runtime of
    this function is O(2^2n).

    Args:
        operator: a list of lists (rows) representing a 2^n by 2^n
            matrix.

    Returns:
        A QubitOperator instance corresponding to the expansion of
        the input operator as a sum of Pauli strings:

        O = 2^-n \sum_P tr(O*P) P
    """

    nrows = len(operator)
    ncols = len(operator[0])

    # Check if the input operator is square
    if nrows != ncols:
        raise Exception("The input operator is not square")

    # Check if the dimensions are powers of 2
    if not (((nrows & (nrows - 1)) == 0) and nrows > 0):
        raise Exception("The number of rows is not a power of 2")
    if not (((ncols & (ncols - 1)) == 0) and ncols > 0):
        raise Exception("The number of cols is not a power of 2")

    n = int(np.log2(nrows))  # number of qubits

    def decode(bit_string):  # Helper function for converting any 2n-bit
        # string to a label vector representing a Pauli
        # string of length n

        if len(bit_string) != 2 * n:
            raise Exception("LH_expand:decode: input bit string length not 2n")

        output_label = list(np.zeros(n))
        for i in range(0, n):
            output_label[i] = bin2dec(bit_string[2 * i : 2 * i + 2])

        return output_label

    def trace_product(label_vec):  # Helper function for computing tr(OP)
        # where O is the input operator and P is a
        # Pauli string operator

        def f(j):  # Function which computes the index of the nonzero
            # element in P for a given column j

            j_str = dec2bin(j, n)
            for index in range(0, n):
                if label_vec[index] in [1, 2]:  # flip if X or Y
                    j_str[index] = int(not j_str[index])
            return bin2dec(j_str)

        def nz(j):  # Function which computes the value of the nonzero
            # element in P on the column j

            val_nz = 1.0
            j_str = dec2bin(j, n)
            for index in range(0, n):
                if label_vec[index] == 2:
                    if j_str[index] == 0:
                        val_nz = val_nz * (1j)
                    if j_str[index] == 1:
                        val_nz = val_nz * (-1j)
                if label_vec[index] == 3:
                    if j_str[index] == 1:
                        val_nz = val_nz * (-1)
            return val_nz

        # Compute the trace
        tr = 0.0
        for j in range(0, 2 ** n):  # loop over the columns
            tr = tr + operator[j][f(j)] * nz(j)

        return tr / 2 ** n

    # Expand the operator in Pauli basis
    coeffs = list(np.zeros(4 ** n))
    labels = list(np.zeros(4 ** n))
    for i in range(0, 4 ** n):  # loop over all 2n-bit strings
        current_string = dec2bin(i, 2 * n)  # see util.py
        current_label = decode(current_string)
        coeffs[i] = trace_product(current_label)
        labels[i] = current_label

    return get_qubitop_from_coeffs_and_labels(coeffs, labels)


def get_qubitop_from_coeffs_and_labels(
    coeffs: List[float], labels: List[List[int]]
) -> QubitOperator:
    """Generates a QubitOperator based on a coefficient vector and
    a label matrix.

    Args:
        coeffs: a list of floats representing the coefficients
            for the terms in the Hamiltonian
        labels: a list of lists (a matrix) where each list
            is a vector of integers representing the Pauli
            string. See pauliutil.py for details.

    Example:

        The Hamiltonian H = 0.1 X1 X2 - 0.4 Y1 Y2 Z3 Z4 can be
        initiated by calling

        H = QubitOperator([0.1, -0.4],\    # coefficients
                    [[1 1 0 0],\  # label matrix
                        [2 2 3 3]])
    """

    output = QubitOperator()
    for i in range(0, len(labels)):
        string_term = ""
        for ind, elem in enumerate(labels[i]):
            pauli_symbol = ""
            if elem == 1:
                pauli_symbol = "X" + str(ind) + " "
            if elem == 2:
                pauli_symbol = "Y" + str(ind) + " "
            if elem == 3:
                pauli_symbol = "Z" + str(ind) + " "
            string_term += pauli_symbol

        output += coeffs[i] * QubitOperator(string_term)

    return output


def generate_random_qubitop(
    nqubits: int,
    nterms: int,
    nlocality: int,
    max_coeff: float,
    fixed_coeff: bool = False,
) -> QubitOperator:
    """Generates a Hamiltonian with term coefficients uniformly distributed
    in [-max_coeff, max_coeff].

    Args:
        nqubits - number of qubits
        nterms    - number of terms in the Hamiltonian
        nlocality - locality of the Hamiltonian
        max_coeff - bound for generating the term coefficients
        fixed_coeff (bool) - If true, all the terms are assign the
            max_coeff as coefficient.

    Returns:
        A QubitOperator with the appropriate coefficient vector
        and label matrix.
    """
    # generate random coefficient vector
    if fixed_coeff:
        coeffs = [max_coeff] * nterms
    else:
        coeffs = list(np.zeros(nterms))
        for j in range(0, nterms):
            coeffs[j] = random.uniform(-max_coeff, max_coeff)

    # generate random label vector
    labels = list(np.zeros(nterms, dtype=int))
    label_set = set()
    j = 0
    while j < nterms:
        inds_nontrivial = sorted(random.sample(range(0, nqubits), nlocality))
        label = list(np.zeros(nqubits, dtype=int))
        for ind in inds_nontrivial:
            label[ind] = random.randint(1, 3)
        if str(label) not in label_set:
            labels[j] = label
            j += 1
            label_set.add(str(label))
    return get_qubitop_from_coeffs_and_labels(coeffs, labels)


def evaluate_qubit_operator(
    qubit_operator: QubitOperator, expectation_values: ExpectationValues
) -> ValueEstimate:
    """Evaluate the expectation value of a qubit operator using
    expectation values for the terms.

    Args:
        qubit_operator (openfermion.QubitOperator): the operator
        expectation_values (core.measurement.ExpectationValues): the expectation values

    Returns:
        value_estimate (zquantum.core.utils.ValueEstimate): stores the value of the expectation and its
             precision
    """

    # Sum the contributions from all terms
    total = 0

    # Add all non-trivial terms
    term_index = 0
    for term in qubit_operator.terms:
        total += np.real(
            qubit_operator.terms[term] * expectation_values.values[term_index]
        )
        term_index += 1

    value_estimate = ValueEstimate(total)
    return value_estimate


def evaluate_operator_for_parameter_grid(
    ansatz, grid, backend, operator, previous_layer_params=[]
):
    """Evaluate the expectation value of an operator for every set of circuit
    parameters in the parameter grid.

    Args:
        ansatz (dict): the ansatz
        grid (zquantum.core.circuit.ParameterGrid): The parameter grid containing
            the parameters for the last layer of the ansatz
        backend (zquantum.core.interfaces.backend.QuantumSimulator): the backend 
            to run the circuits on 
        operator (openfermion.ops.QubitOperator): the operator
        previous_layer_params (array): A list of the parameters for previous layers
            of the ansatz

    Returns:
        value_estimate (zquantum.core.utils.ValueEstimate): stores the value of the expectation and its
             precision
        optimal_parameters (numpy array): the ansatz parameters representing the ansatz parameters 
            resulting in the best minimum evaluation. If multiple sets of parameters evaluate to the same value, 
            the first set of parameters is chosen as the optimal.
    """
    parameter_grid_evaluation = []
    circuitset = []
    params_set = []
    for last_layer_params in grid.params_list:
        # Build the ansatz circuit
        params = np.concatenate(
            (np.asarray(previous_layer_params), np.asarray(last_layer_params))
        )

        # Build the ansatz circuit
        circuitset.append(ansatz.get_executable_circuit(params))
        params_set.append(params)

    expectation_values_set = backend.get_expectation_values_for_circuitset(
        circuitset, operator
    )

    min_value_estimate = None
    for params, expectation_values in zip(params_set, expectation_values_set):
        expectation_values = expectation_values_to_real(expectation_values)
        value_estimate = ValueEstimate(sum(expectation_values.values))
        parameter_grid_evaluation.append(
            {
                "value": value_estimate,
                "parameter1": params[-2],
                "parameter2": params[-1],
            }
        )

        if min_value_estimate is None:
            min_value_estimate = value_estimate
            optimal_parameters = params
        elif value_estimate.value < min_value_estimate.value:
            min_value_estimate = value_estimate
            optimal_parameters = params

    return parameter_grid_evaluation, optimal_parameters


def reverse_qubit_order(qubit_operator: QubitOperator, n_qubits: Optional[int] = None):
    """Reverse the order of qubit indices in a qubit operator.

    Args:
        qubit_operator (openfermion.QubitOperator): the operator
        n_qubits (int): total number of qubits. Needs to be provided when 
                    the size of the system of interest is greater than the size of qubit operator (optional)

    Returns:
        reversed_op (openfermion.ops.QubitOperator)
    """

    reversed_op = QubitOperator()

    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    if n_qubits < count_qubits(qubit_operator):
        raise ValueError("Invalid number of qubits specified.")

    for term in qubit_operator.terms:
        new_term = []
        for factor in term:
            new_factor = list(factor)
            new_factor[0] = n_qubits - 1 - new_factor[0]
            new_term.append(tuple(new_factor))
        reversed_op += QubitOperator(tuple(new_term), qubit_operator.terms[term])
    return reversed_op


def expectation(qubit_op, wavefunction, reverse_operator=True):
    """Get the expectation value of a qubit operator with respect to a wavefunction.
    Args:
        qubit_op (openfermion.ops.QubitOperator): the operator
        wavefunction (pyquil.wavefunction.Wavefunction): the wavefunction
        reverse_operator (boolean): whether to reverse order of qubit operator
            before computing expectation value. This should be True if the convention
            of the basis states used for the wavefunction is the opposite of the one in
            the qubit operator. This is the case, e.g. when the wavefunction comes from
            Pyquil.
    Returns:
        complex: the expectation value
    """
    n_qubits = wavefunction.amplitudes.shape[0].bit_length() - 1

    # Convert the qubit operator to a sparse matrix. Note that the qubit indices
    # must be reversed because OpenFermion and pyquil use different conventions
    # for how to order the computational basis states!
    if reverse_operator:
        qubit_op = reverse_qubit_order(qubit_op, n_qubits=n_qubits)
    sparse_op = get_sparse_operator(qubit_op, n_qubits=n_qubits)

    # Computer the expectation value
    exp_val = openfermion_expectation(sparse_op, wavefunction.amplitudes)
    return exp_val


def change_operator_type(operator, operatorType):
    """Take an operator and attempt to cast it to an operator of a different type

    Args:
        operator: The operator
        operatorType: The type of the operator that the original operator is
            cast to
    Returns:
        An operator with type operatorType
    """
    new_operator = operatorType()
    for op in operator.terms:
        new_operator += operatorType(tuple(op), operator.terms[op])

    return new_operator


def get_fermion_number_operator(n_qubits, n_particles=None):
    """Return a FermionOperator representing the number operator
    for n qubits.
    If `n_particles` is specified, it can be used for creating constraint on the number of particles.

    Args:
        n_qubits (int): number of qubits in the system
        n_particles (int): number of particles in the system.
            If specified, it is substracted from the number
            operator such as expectation value is zero.
    Returns:
         (openfermion.ops.FermionOperator): the number operator
    """
    operator = number_operator(n_qubits)
    if n_particles is not None:
        operator += FermionOperator("", -1.0 * float(n_particles))
    return get_interaction_operator(operator)


def get_diagonal_component(operator):
    if isinstance(operator, InteractionOperator):
        return _get_diagonal_component_interaction_operator(operator)
    elif isinstance(operator, PolynomialTensor):
        return _get_diagonal_component_polynomial_tensor(operator)
    else:
        raise TypeError(
            f"Getting diagonal component not supported for {0}".format(type(operator))
        )


def _get_diagonal_component_polynomial_tensor(polynomial_tensor):
    """Get the component of an interaction operator that is
    diagonal in the computational basis under Jordan-Wigner
    transformation (i.e., the terms that can be expressed
    as products of number operators).
    Args:
        interaction_operator (openfermion.ops.InteractionOperator): the operator
    
    Returns:
        tuple: two openfermion.ops.InteractionOperator objects. The first is the
            diagonal component, and the second is the remainder.
    """
    n_modes = count_qubits(polynomial_tensor)
    remainder_tensors = {}
    diagonal_tensors = {}

    diagonal_tensors[()] = polynomial_tensor.constant
    for key in polynomial_tensor.n_body_tensors:
        if key == ():
            continue
        remainder_tensors[key] = np.zeros((n_modes,) * len(key), complex)
        diagonal_tensors[key] = np.zeros((n_modes,) * len(key), complex)

        for indices in itertools.product(range(n_modes), repeat=len(key)):
            creation_counts = {}
            annihilation_counts = {}

            for meta_index, index in enumerate(indices):
                if key[meta_index] == 0:
                    if annihilation_counts.get(index) is None:
                        annihilation_counts[index] = 1
                    else:
                        annihilation_counts[index] += 1
                elif key[meta_index] == 1:
                    if creation_counts.get(index) is None:
                        creation_counts[index] = 1
                    else:
                        creation_counts[index] += 1

            term_is_diagonal = True
            for index in creation_counts:
                if creation_counts[index] != annihilation_counts.get(index):
                    term_is_diagonal = False
                    break
            if term_is_diagonal:
                for index in annihilation_counts:
                    if annihilation_counts[index] != creation_counts.get(index):
                        term_is_diagonal = False
                        break
            if term_is_diagonal:
                diagonal_tensors[key][indices] = polynomial_tensor.n_body_tensors[key][
                    indices
                ]
            else:
                remainder_tensors[key][indices] = polynomial_tensor.n_body_tensors[key][
                    indices
                ]

    return PolynomialTensor(diagonal_tensors), PolynomialTensor(remainder_tensors)


def _get_diagonal_component_interaction_operator(interaction_operator):
    """Get the component of an interaction operator that is
    diagonal in the computational basis under Jordan-Wigner
    transformation (i.e., the terms that can be expressed
    as products of number operators).
    Args:
        interaction_operator (openfermion.ops.InteractionOperator): the operator
    
    Returns:
        tuple: two openfermion.ops.InteractionOperator objects. The first is the
            diagonal component, and the second is the remainder.
    """

    one_body_tensor = np.zeros(
        interaction_operator.one_body_tensor.shape, dtype=complex
    )
    two_body_tensor = np.zeros(
        interaction_operator.two_body_tensor.shape, dtype=complex
    )
    diagonal_op = InteractionOperator(
        interaction_operator.constant, one_body_tensor, two_body_tensor
    )

    one_body_tensor = np.copy(interaction_operator.one_body_tensor).astype(complex)
    two_body_tensor = np.copy(interaction_operator.two_body_tensor).astype(complex)
    remainder_op = InteractionOperator(0.0, one_body_tensor, two_body_tensor)

    n_spin_orbitals = interaction_operator.two_body_tensor.shape[0]

    for p in range(n_spin_orbitals):
        for q in range(n_spin_orbitals):
            diagonal_op.two_body_tensor[
                p, q, p, q
            ] = interaction_operator.two_body_tensor[p, q, p, q]
            diagonal_op.two_body_tensor[
                p, q, q, p
            ] = interaction_operator.two_body_tensor[p, q, q, p]
            remainder_op.two_body_tensor[p, q, p, q] = 0.0
            remainder_op.two_body_tensor[p, q, q, p] = 0.0

    for p in range(n_spin_orbitals):
        diagonal_op.one_body_tensor[p, p] = interaction_operator.one_body_tensor[p, p]
        remainder_op.one_body_tensor[p, p] = 0.0

    return diagonal_op, remainder_op


def get_polynomial_tensor(fermion_operator, n_qubits=None):
    r"""Convert a fermionic operator to a Polynomial Tensor.

    Args:
        fermion_operator (openferion.ops.FermionOperator): The operator.
        n_qubits (int): The number of qubits to be included in the
            PolynomialTensor. Must be at least equal to the number of qubits
            that are acted on by fermion_operator. If None, then the number of
            qubits is inferred from fermion_operator.
    
    Returns:
        openfermion.ops.PolynomialTensor: The tensor representation of the
            operator.
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError("Input must be a FermionOperator.")

    if n_qubits is None:
        n_qubits = count_qubits(fermion_operator)
    if n_qubits < count_qubits(fermion_operator):
        raise ValueError("Invalid number of qubits specified.")

    # Normal order the terms and initialize.
    fermion_operator = normal_ordered(fermion_operator)
    tensor_dict = {}

    # Loop through terms and assign to matrix.
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]

        # Handle constant shift.
        if len(term) == 0:
            tensor_dict[()] = coefficient

        else:
            key = tuple([operator[1] for operator in term])
            if tensor_dict.get(key) is None:
                tensor_dict[key] = np.zeros((n_qubits,) * len(key), complex)

            indices = tuple([operator[0] for operator in term])
            tensor_dict[key][indices] = coefficient

    return PolynomialTensor(tensor_dict)


def qubitop_to_paulisum(
    qubit_operator: QubitOperator,
    qubits: Union[List[cirq.GridQubit], List[cirq.LineQubit]] = None,
) -> cirq.PauliSum:
    """Convert and openfermion QubitOperator to a cirq PauliSum

    Args:
        qubit_operator (openfermion.QubitOperator): The openfermion operator to convert
        qubits()

    Returns:
        cirq.PauliSum 
    """
    operator_map = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

    if qubits is None:
        qubits = [cirq.GridQubit(i, 0) for i in range(count_qubits(qubit_operator))]

    converted_sum = cirq.PauliSum()
    for term, coefficient in qubit_operator.terms.items():

        # Identity term
        if len(term) == 0:
            converted_sum += coefficient
            continue

        cirq_term = cirq.PauliString()
        for qubit_index, operator in term:
            cirq_term *= operator_map[operator](qubits[qubit_index])
        converted_sum += cirq_term * coefficient

    return converted_sum

