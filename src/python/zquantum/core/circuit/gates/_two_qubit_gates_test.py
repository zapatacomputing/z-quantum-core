"""Test cases for two qubit gates."""
import pytest
import sympy

from . import CustomGate
from ._two_qubit_gates import XX, YY, ZZ


@pytest.mark.parametrize(
    "angle,expected_matrix",
    [
        (0, sympy.eye(4)),
        (
            sympy.pi,
            sympy.Matrix([[0, 0, 0, -1j], [0, 0, -1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]]),
        ),
        (
            sympy.pi / 2,
            sympy.Matrix([[1, 0, 0, -1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [-1j, 0, 0, 1]])
            / sympy.sqrt(2),
        ),
    ],
)
def test_XX_matrix_is_correct_for_characteristic_values_of_angle(
    angle, expected_matrix
):
    assert XX(qubits=(0, 1), angle=angle) == CustomGate(matrix=expected_matrix, qubits=(0, 1))


@pytest.mark.parametrize(
    "angle,expected_matrix",
    [
        (0, sympy.eye(4)),
        (
            sympy.pi,
            sympy.Matrix([[0, 0, 0, 1j], [0, 0, -1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]]),
        ),
        (
            sympy.pi / 2,
            sympy.Matrix([[1, 0, 0, 1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [1j, 0, 0, 1]])
            / sympy.sqrt(2),
        ),
    ],
)
def test_YY_matrix_is_correct_for_characteristic_values_of_angle(
    angle, expected_matrix
):
    assert YY(qubits=(0, 1), angle=angle) == CustomGate(matrix=expected_matrix, qubits=(0, 1))


@pytest.mark.parametrize(
    "angle,expected_matrix",
    [
        (0, sympy.eye(4)),
        (2 * sympy.pi, -sympy.eye(4)),
        (
            sympy.pi,
            sympy.Matrix([[-1j, 0, 0, 0], [0, 1j, 0, 0], [0, 0, 1j, 0], [0, 0, 0, -1j]]),
        ),
        (
            sympy.pi / 2,
            (
                sympy.Matrix(
                    [
                        [1 - 1j, 0, 0, 0],
                        [0, 1 + 1j, 0, 0],
                        [0, 0, 1 + 1j, 0],
                        [0, 0, 0, 1 - 1j],
                    ]
                )
                / sympy.sqrt(2)
            ),
        ),
    ],
)
def test_ZZ_matrix_is_correct_for_characteristic_values_of_angle(
    angle, expected_matrix
):
    assert ZZ(qubits=(0, 1), angle=angle) == CustomGate(matrix=expected_matrix, qubits=(0, 1))
