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
    assert XX(0, 1, angle=angle) == CustomGate(matrix=expected_matrix, qubits=(0, 1))


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
    assert YY(0, 1, angle=angle) == CustomGate(matrix=expected_matrix, qubits=(0, 1))


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
    assert ZZ(0, 1, angle=angle) == CustomGate(matrix=expected_matrix, qubits=(0, 1))


@pytest.mark.parametrize("gate_cls", [SWAP, CZ, CNOT])
def test_nonparametric_two_qubit_gates_have_no_params(gate_cls):
    assert gate_cls(0, 3).params == ()


@pytest.mark.parametrize("gate_cls", [XX, YY, ZZ, CPHASE])
@pytest.mark.parametrize("angle", [sympy.Symbol("alpha"), np.pi / 2])
def test_rotation_gates_have_a_single_parameter_equal_to_their_angle(gate_cls, angle):
    assert gate_cls(1, 4, angle).params == (angle,)


class TestStringRepresentationOfTwoQubitGates:

    @pytest.mark.parametrize(
        "gate, expected_representation",
        [
            (SWAP(0, 1), "SWAP(0, 1)"),
            (CZ(4, 2), "CZ(4, 2)"),
            (CNOT(1, 3), "CNOT(1, 3)")
        ]
    )
    def test_representation_of_two_qubit_nonparametric_gates_looks_like_initiizer_call(
        self, gate, expected_representation
    ):
        assert str(gate) == repr(gate) == expected_representation


    @pytest.mark.parametrize(
        "gate, expected_representation",
        [
            (XX(0, 1), "XX(0, 1, angle=theta)"),
            (XX(1, 2, sympy.pi), "XX(1, 2, angle=pi)"),
            (YY(0, 1, sympy.pi / 2), "YY(0, 1, angle=pi/2)"),
            (YY(3, 1, 0.1), f"YY(3, 1, angle={0.1})"),
            (ZZ(4, 0, sympy.Symbol("x") + sympy.Symbol("y")), "ZZ(4, 0, angle=x+y)"),
            (ZZ(0, 1, np.pi / 4), f"ZZ(0, 1, angle={np.pi/4})"),
            (CPHASE(0, 1, np.pi / 5), f"CPHASE(0, 1, angle={np.pi/5})"),
            (CPHASE(4, 1, sympy.cos(sympy.Symbol("x"))), f"CPHASE(4, 1, angle=cos(x))")
        ]
    )
    def test_representation_of_two_qubit_rotations_looks_like_initializer_call_with_keyword_angle(
        self, gate, expected_representation
    ):
        assert str(gate) == repr(gate) == expected_representation
