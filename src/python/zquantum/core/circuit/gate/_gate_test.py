import numpy as np
import pytest
import json
import os
import sympy
from ...utils import SCHEMA_VERSION

from ._gate import Gate

#### __init__ ####
@pytest.mark.parametrize(
    "matrix",
    [
        sympy.Matrix([[0, 0], [0, 0], [0, 0]]),
        sympy.Matrix([[0], [0]]),
        sympy.Matrix([[0, 0, 0], [0, 0, 0]]),
        sympy.Matrix([[0]]),
    ],
)
def test_creating_gate_with_rectangular_matrix_fails(matrix):
    """The Gate class should raise an assertion error if the matrix is not square"""
    qubits = (0,)
    with pytest.raises(AssertionError):
        Gate(matrix, qubits)


@pytest.mark.parametrize(
    "matrix",
    [
        sympy.Matrix([[], []]),
        sympy.Matrix([[0], [0]]),
        sympy.Matrix([[0, 0, 0], [0, 0, 0]]),
        sympy.Matrix([[0, 0, 0, 0], [0, 0, 0, 0]]),
    ],
)
def test_creating_gate_with_invalid_matrix_size_for_one_qubit_fails(matrix):
    """The Gate class should raise an assertion error if the matrix is not 2x2"""
    qubits = (0,)
    with pytest.raises(AssertionError):
        Gate(matrix, qubits)


@pytest.mark.parametrize(
    "matrix",
    [
        sympy.Matrix([[], []]),
        sympy.Matrix([[0], [0]]),
        sympy.Matrix([[0, 0], [0, 0]]),
        sympy.Matrix([[0, 0, 0], [0, 0, 0]]),
    ],
)
def test_creating_gate_with_invalid_matrix_size_for_two_qubit_fails(matrix):
    """The Gate class should raise an assertion error if the matrix is not 4x4"""
    qubits = (
        0,
        1,
    )
    with pytest.raises(AssertionError):
        Gate(matrix, qubits)


@pytest.mark.parametrize(
    "qubits", [(0, 0), (0, 1, 1), (0, 1, 0), (0, 1, 2, 3, 4, 5, 6, 7, 3),]
)
def test_creating_gate_with_repeated_qubits_fails(qubits):
    """The Gate class should raise an assertion error if all qubit indices are not unique"""
    matrix = sympy.Matrix(np.eye(2 ** len(qubits)))
    with pytest.raises(AssertionError):
        Gate(matrix, qubits)


@pytest.mark.parametrize("number_of_qubits", [1, 2, 3, 4, 5, 6])
def test_creating_identity_gate_succeeds(number_of_qubits):
    """The Gate class should be able to handle the identity gate for different numbers of qubits"""
    # Given
    qubits = tuple([i for i in range(number_of_qubits)])
    matrix = sympy.Matrix(np.eye(2 ** number_of_qubits))

    # When
    gate = Gate(matrix, qubits)

    # Then
    assert gate.qubits == qubits
    assert all(
        element_gate == element for element_gate, element in zip(gate.matrix, matrix)
    )


def test_creating_complex_gate():
    """The Gate class should be able to handle complex matrices"""
    # Given
    qubits = (0,)
    matrix = sympy.Matrix(
        [[complex(0), complex(0 + 1j)], [complex(0 - 1j), complex(0)],]
    )

    # When
    gate = Gate(matrix, qubits)

    # Then
    assert gate.qubits == qubits
    assert all(
        element_gate == element for element_gate, element in zip(gate.matrix, matrix)
    )


def test_creating_gate_support_non_unitary_gates():
    """The Gate class should be able to handle non unitary gates"""
    # Given
    qubits = (
        0,
        1,
    )
    matrix = sympy.Matrix(np.zeros(shape=(2 ** len(qubits), 2 ** len(qubits))))

    # When
    gate = Gate(matrix, qubits)

    # Then
    assert gate.qubits == qubits
    assert all(element == 0 for element in gate.matrix)


def test_creating_parameterized_gate_repeated_symbol():
    """The Gate class should be able to handle complex matrices"""
    # Given
    qubits = (0,)
    matrix = sympy.Matrix(np.eye(2 ** len(qubits)))
    matrix[0, 0] = sympy.Symbol("theta_0")
    matrix[1, 1] = 3 * sympy.Symbol("theta_0")

    # When
    gate = Gate(matrix, qubits)

    # Then
    assert gate.qubits == qubits
    assert all(
        gate_element == element for gate_element, element in zip(gate.matrix, matrix)
    )
    assert gate.symbolic_params == set([sympy.Symbol("theta_0")])


def test_creating_parameterized_gate_multiple_symbols():
    """The Gate class should be able to handle complex matrices"""
    # Given
    qubits = (0,)
    matrix = sympy.Matrix(np.eye(2 ** len(qubits)))
    matrix[0, 0] = sympy.Symbol("theta_0")
    matrix[1, 1] = 3 * sympy.Symbol("theta_1")

    # When
    gate = Gate(matrix, qubits)

    # Then
    assert gate.qubits == qubits
    assert all(
        gate_element == element for gate_element, element in zip(gate.matrix, matrix)
    )
    assert gate.symbolic_params == set(
        [sympy.Symbol("theta_0"), sympy.Symbol("theta_1")]
    )


#### __eq__ ####
@pytest.mark.parametrize(
    "matrix, qubits",
    [
        [sympy.Matrix(np.eye(2)), (0,)],
        [sympy.Matrix(np.eye(2)), (1,)],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (0,),],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (1,),],
        [
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]),
            (0, 2),
        ],
    ],
)
def test_gate_eq_same_gates(matrix, qubits):
    """The Gate class should be able to be able to compare two gates"""
    # Given
    gate = Gate(matrix, qubits)
    another_gate = Gate(matrix, qubits)

    # When
    are_equal = gate == another_gate

    # Then
    assert are_equal


@pytest.mark.parametrize(
    "matrix1, qubits1, matrix2, qubits2",
    [
        [sympy.Matrix(np.eye(2)), (0,), sympy.Matrix(np.eye(2)), (1,),],
        [
            sympy.Matrix(np.eye(2)),
            (1,),
            sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]),
            (0,),
        ],
        [
            sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]),
            (0,),
            sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]),
            (1,),
        ],
        [
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]),
            (0, 2),
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],]),
            (0, 2),
        ],
    ],
)
def test_gate_eq_not_same_gates(matrix1, qubits1, matrix2, qubits2):
    """The Gate class should be able to be able to compare two gates"""
    # Given
    gate = Gate(matrix1, qubits1)
    another_gate = Gate(matrix2, qubits2)

    # When
    are_equal = gate == another_gate

    # Then
    assert not are_equal


#### __repr___ ####
@pytest.mark.parametrize(
    "matrix, qubits",
    [
        [sympy.Matrix(np.eye(2)), (0,)],
        [sympy.Matrix(np.eye(2)), (1,)],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (0,),],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (1,),],
        [
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]),
            (0, 2),
        ],
    ],
)
def test_gate__repr__(matrix, qubits):
    """The Gate class's __repr__ method is as expected"""
    # Given
    gate = Gate(matrix, qubits)

    # When
    representation = gate.__repr__()

    # Then
    assert "zquantum.core.circuit.gate.Gate" in representation
    assert "matrix" in representation
    assert "qubits" in representation


#### to_dict ####
@pytest.mark.parametrize(
    "matrix, qubits",
    [
        [sympy.Matrix(np.eye(2)), (0,)],
        [sympy.Matrix(np.eye(2)), (1,)],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (0,),],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (1,),],
        [
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]),
            (0, 2),
        ],
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
    assert gate_dict["matrix"] == matrix


@pytest.mark.parametrize(
    "matrix, qubits",
    [
        [sympy.Matrix(np.eye(2)), (0,)],
        [sympy.Matrix(np.eye(2)), (1,)],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (0,),],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (1,),],
        [
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]),
            (0, 2),
        ],
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
            assert float(real_value) == sympy.re(matrix[row_index, col_index])
        for col_index, imag_value in enumerate(row["imag"]):
            assert float(imag_value) == sympy.im(matrix[row_index, col_index])


#### save ####
@pytest.mark.parametrize(
    "matrix, qubits",
    [
        [sympy.Matrix(np.eye(2)), (0,)],
        [sympy.Matrix(np.eye(2)), (1,)],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (0,),],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (1,),],
        [
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]),
            (0, 2),
        ],
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
            assert float(real_value) == matrix[row_index, col_index].as_real_imag()[0]
        for col_index, imag_value in enumerate(row["imag"]):
            assert float(imag_value) == matrix[row_index, col_index].as_real_imag()[1]

    # os.remove("gate.json")


#### load ####
@pytest.mark.parametrize(
    "matrix, qubits",
    [
        [sympy.Matrix(np.eye(2)), (0,)],
        [sympy.Matrix(np.eye(2)), (1,)],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (0,),],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (1,),],
        [
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]),
            (0, 2),
        ],
    ],
)
def test_gate_is_successfully_loaded_from_a_file(matrix, qubits):
    """The Gate class should be able to be loaded from a file"""
    # Given
    gate = Gate(matrix, qubits)

    gate.save("gate.json")

    # When
    new_gate = Gate.load("gate.json")

    # Then
    assert gate == new_gate

    os.remove("gate.json")


@pytest.mark.parametrize(
    "matrix, qubits",
    [
        [sympy.Matrix(np.eye(2)), (0,)],
        [sympy.Matrix(np.eye(2)), (1,)],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (0,),],
        [sympy.Matrix([[0, 0 + 1j], [0 - 1j, 0],]), (1,),],
        [
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]),
            (0, 2),
        ],
    ],
)
def test_gate_is_successfully_loaded_from_a_dict(matrix, qubits):
    """The Gate class should be able to be loaded from a dict"""
    # Given
    gate = Gate(matrix, qubits)

    gate_dict = gate.to_dict(serializable=True)

    # When
    new_gate = Gate.load(gate_dict)

    # Then
    assert gate == new_gate

