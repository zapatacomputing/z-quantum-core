import numpy as np
import pytest
import json
import os
from ...utils import SCHEMA_VERSION

from ._gate import (
    Gate,
)

#### __init__ ####
@pytest.mark.parametrize(
    "matrix", [
        np.asarray([np.asarray([0,0]), np.asarray([0,0, 0])]),
        np.asarray([np.asarray([0,0]), np.asarray([0])]),
        np.asarray([np.asarray([]), np.asarray([0,0, 0])]),
        np.asarray([np.asarray([0])]),
    ]
)
def test_creating_gate_with_rectangular_matrix_fails(matrix):
    """The Gate class should raise an assertion error if the matrix is not square"""
    qubits = (0,)
    with pytest.raises(AssertionError):
        Gate(matrix, qubits)

@pytest.mark.parametrize(
    "matrix", [
        np.asarray([np.asarray([]), np.asarray([])]),
        np.asarray([np.asarray([0]), np.asarray([0])]),
        np.asarray([np.asarray([0,0,0]), np.asarray([0,0,0])]),
        np.asarray([np.asarray([0,0,0,0]), np.asarray([0,0,0,0])]),
    ]
)
def test_creating_gate_with_invalid_matrix_size_for_one_qubit_fails(matrix):
    """The Gate class should raise an assertion error if the matrix is not 2x2"""
    qubits = (0,)
    with pytest.raises(AssertionError):
        Gate(matrix, qubits)

@pytest.mark.parametrize(
    "matrix", [
        np.asarray([np.asarray([]), np.asarray([])]),
        np.asarray([np.asarray([0]), np.asarray([0])]),
        np.asarray([np.asarray([0,0]), np.asarray([0,0])]),
        np.asarray([np.asarray([0,0,0]), np.asarray([0,0,0])]),
    ]
)
def test_creating_gate_with_invalid_matrix_size_for_two_qubit_fails(matrix):
    """The Gate class should raise an assertion error if the matrix is not 4x4"""
    qubits = (0,1,)
    with pytest.raises(AssertionError):
        Gate(matrix, qubits)

@pytest.mark.parametrize(
    "qubits", [
        (0,0),
        (0,1,1),
        (0,1,0),
        (0,1,2,3,4,5,6,7,3),
    ]
)
def test_creating_gate_with_repeated_qubits_fails(qubits):
    """The Gate class should raise an assertion error if all qubit indices are not unique"""
    matrix = np.eye(2**len(qubits))
    with pytest.raises(AssertionError):
        Gate(matrix, qubits)

@pytest.mark.parametrize(
    "number_of_qubits", [1,2,3,4,5,6]
)
def test_creating_identity_gate_for_succeeds(number_of_qubits):
    """The Gate class should be able to handle the identity gate for different numbers of qubits"""
    # Given
    qubits = tuple([i for i in range(number_of_qubits)])
    matrix = np.eye(2**number_of_qubits)

    # When
    gate = Gate(matrix, qubits)

    # Then
    assert gate.qubits == qubits
    assert all(element_gate == element for gate_row, row in zip(gate.matrix, matrix) for element_gate, element in zip(gate_row, row))

def test_creating_complex_gate():
    """The Gate class should be able to handle complex matrices"""
    # Given
    qubits = (0,1,)
    matrix = np.eye(2**len(qubits), dtype=complex)

    # When
    gate = Gate(matrix, qubits)

    # Then
    assert gate.qubits == qubits
    assert all(element_gate == element for gate_row, row in zip(gate.matrix, matrix) for element_gate, element in zip(gate_row, row))

def test_creating_gate_support_non_unitary_gates():
    """The Gate class should be able to handle non unitary gates"""
    # Given
    qubits = (0,1,)
    matrix = np.zeros(shape=(2**len(qubits), 2**len(qubits)))

    # When
    gate = Gate(matrix, qubits)

    # Then
    assert gate.qubits == qubits
    assert all(element == 0 for row in gate.matrix for element in row)


#### to_dict ####
@pytest.mark.parametrize(
    "matrix, qubits", [
        [np.eye(2, dtype=complex),(0,)],
        [np.eye(2, dtype=complex),(1,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(0,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(1,)],
        [np.asarray([np.asarray([1, 0, 0, 0], dtype=complex), np.asarray([0, 1, 0, 0], dtype=complex), np.asarray([0, 0, 0, 1], dtype=complex), np.asarray([0, 0, 1, 0], dtype=complex)]),(0,2)],
    ],
)
def test_gate_is_successfully_converted_to_dict_form(matrix, qubits):
    """The Gate class should be able to be converted to a dict"""
    # Given
    gate = Gate(matrix, qubits)

    # When
    gate_dict = gate.to_dict(serializable=False)

    # Then
    assert gate_dict["schema"] == SCHEMA_VERSION + "-gate"
    assert gate_dict["qubits"] == qubits
    assert all(element_dict == element for row_dict, row in zip(gate_dict["matrix"], matrix) for element_dict, element in zip(row_dict, row))

@pytest.mark.parametrize(
    "matrix, qubits", [
        [np.eye(2, dtype=complex),(0,)],
        [np.eye(2, dtype=complex),(1,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(0,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(1,)],
        [np.asarray([np.asarray([1, 0, 0, 0], dtype=complex), np.asarray([0, 1, 0, 0], dtype=complex), np.asarray([0, 0, 0, 1], dtype=complex), np.asarray([0, 0, 1, 0], dtype=complex)]),(0,2)],
    ],
)
def test_gate_is_successfully_converted_to_serializable_dict_form(matrix, qubits):
    """The Gate class should be able to be converted to a dict in a serializable form"""
    # Given
    gate = Gate(matrix, qubits)

    # When
    gate_dict = gate.to_dict(serializable=True)

    # Then
    assert gate_dict["schema"] == SCHEMA_VERSION + "-gate"
    assert gate_dict["qubits"] == list(qubits)
    for row_index, row in enumerate(gate_dict["matrix"]):
        for col_index, real_value in enumerate(row["real"]):
            assert real_value == matrix[row_index][col_index].real 
        for col_index, imag_value in enumerate(row["imag"]):
            assert imag_value == matrix[row_index][col_index].imag 

#### save ####
@pytest.mark.parametrize(
    "matrix, qubits", [
        [np.eye(2, dtype=complex),(0,)],
        [np.eye(2, dtype=complex),(1,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(0,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(1,)],
        [np.asarray([np.asarray([1, 0, 0, 0], dtype=complex), np.asarray([0, 1, 0, 0], dtype=complex), np.asarray([0, 0, 0, 1], dtype=complex), np.asarray([0, 0, 1, 0], dtype=complex)]),(0,2)],
    ],
)
def test_gate_is_successfully_saved_to_a_file(qubits, matrix):
    """The Gate class should be able to be saved to file"""
    # Given
    gate = Gate(matrix, qubits)

    # When
    gate.save("gate.json")
    with open("gate.json", "r") as f:
        saved_data = json.loads(f.read())

    # Then
    assert saved_data["schema"] == SCHEMA_VERSION + "-gate"
    assert saved_data["qubits"] == list(qubits)
    for row_index, row in enumerate(saved_data["matrix"]):
        for col_index, real_value in enumerate(row["real"]):
            assert real_value == matrix[row_index][col_index].real 
        for col_index, imag_value in enumerate(row["imag"]):
            assert imag_value == matrix[row_index][col_index].imag 

    os.remove("gate.json")

#### load ####
@pytest.mark.parametrize(
    "matrix, qubits", [
        [np.eye(2, dtype=complex),(0,)],
        [np.eye(2, dtype=complex),(1,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(0,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(1,)],
        [np.asarray([np.asarray([1, 0, 0, 0], dtype=complex), np.asarray([0, 1, 0, 0], dtype=complex), np.asarray([0, 0, 0, 1], dtype=complex), np.asarray([0, 0, 1, 0], dtype=complex)]),(0,2)],
    ],
)
def test_gate_is_successfully_loaded_from_a_file(matrix, qubits):
    """The Gate class should be able to be loaded from a file"""
    # Given
    gate = Gate(matrix, qubits)

    gate.save("gate.json")

    # When
    gate = Gate.load("gate.json")

    # Then
    assert gate.qubits == qubits
    assert all(element_gate == element for gate_row, row in zip(gate.matrix, matrix) for element_gate, element in zip(gate_row, row))

    os.remove("gate.json")

@pytest.mark.parametrize(
    "matrix, qubits", [
        [np.eye(2, dtype=complex),(0,)],
        [np.eye(2, dtype=complex),(1,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(0,)],
        [np.asarray([np.asarray([0, 0+1j], dtype=complex), np.asarray([0-1j,0], dtype=complex)]),(1,)],
        [np.asarray([np.asarray([1, 0, 0, 0], dtype=complex), np.asarray([0, 1, 0, 0], dtype=complex), np.asarray([0, 0, 0, 1], dtype=complex), np.asarray([0, 0, 1, 0], dtype=complex)]),(0,2)],
    ],
)
def test_gate_is_successfully_loaded_from_a_dict(matrix, qubits):
    """The Gate class should be able to be loaded from a dict"""
    # Given
    gate = Gate(matrix, qubits)

    gate_dict = gate.to_dict(serializable=True)

    # When
    gate = Gate.load(gate_dict)
    
    # Then
    assert gate.qubits == qubits
    assert all(element_gate == element for gate_row, row in zip(gate.matrix, matrix) for element_gate, element in zip(gate_row, row))