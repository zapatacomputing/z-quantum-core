"""Test cases for two qubit gates."""
import numpy as np
import pytest
import sympy

from . import CustomGate
from ._two_qubit_gates import XX, YY, ZZ, SWAP, CNOT, CZ, CPHASE


@pytest.mark.parametrize(
    "angle,expected_matrix",
    [
        (0, sympy.eye(4)),
        (
            sympy.pi,
            sympy.Matrix(
                [[0, 0, 0, -1j], [0, 0, -1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]]
            ),
        ),
        (
            sympy.pi / 2,
            sympy.Matrix(
                [[1, 0, 0, -1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [-1j, 0, 0, 1]]
            )
            / sympy.sqrt(2),
        ),
    ],
)
def test_XX_matrix_is_correct_for_characteristic_values_of_angle(
    angle, expected_matrix
):
    assert XX(0, 1, angle=angle) == CustomGate(
        matrix=expected_matrix, qubits=(0, 1)
    )


@pytest.mark.parametrize(
    "angle,expected_matrix",
    [
        (0, sympy.eye(4)),
        (
            sympy.pi,
            sympy.Matrix(
                [[0, 0, 0, 1j], [0, 0, -1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]]
            ),
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
    assert YY(0, 1, angle=angle) == CustomGate(
        matrix=expected_matrix, qubits=(0, 1)
    )


@pytest.mark.parametrize(
    "angle,expected_matrix",
    [
        (0, sympy.eye(4)),
        (2 * sympy.pi, -sympy.eye(4)),
        (
            sympy.pi,
            sympy.Matrix(
                [[-1j, 0, 0, 0], [0, 1j, 0, 0], [0, 0, 1j, 0], [0, 0, 0, -1j]]
            ),
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
    assert ZZ(0, 1, angle=angle) == CustomGate(
        matrix=expected_matrix, qubits=(0, 1)
    )


@pytest.mark.parametrize("gate_cls", [SWAP, CZ, CNOT])
def test_nonparametric_two_qubit_gates_have_no_params(gate_cls):
    assert gate_cls(0, 3).params == ()


@pytest.mark.parametrize("gate_cls", [XX, YY, ZZ, CPHASE])
@pytest.mark.parametrize("angle", [sympy.Symbol("alpha"), np.pi / 2])
def test_rotation_gates_have_a_single_parameter_equal_to_their_angle(gate_cls, angle):
    assert gate_cls(1, 4, angle).params == (angle,)
