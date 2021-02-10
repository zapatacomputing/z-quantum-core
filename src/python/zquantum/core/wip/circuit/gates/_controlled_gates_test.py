"""Test cases for controlled gates."""
import sympy
import pytest
from ._single_qubit_gates import X, Z
from ._gate import CustomGate, ControlledGate

THETA = sympy.Symbol("theta")
PHI = sympy.Symbol("phi")


@pytest.mark.parametrize(
    "control, target_gate, expected_qubits",
    [
        (1, X(5), (1, 5)),
        (4, Z(0), (4, 0)),
        (
            0,
            CustomGate(
                sympy.Matrix(
                    [
                        [THETA, 0, 0, 0],
                        [0, THETA, 0, 0],
                        [0, 0, 0, -1j * PHI],
                        [0, 0, -1j * PHI, 0],
                    ]
                ),
                (2, 3),
            ),
            (0, 2, 3),
        ),
        (0, ControlledGate(X(2), 1), (0, 1, 2)),
    ],
)
def test_qubits_of_controlled_gate_comprises_control_qubit_and_target_qubits(
    control, target_gate, expected_qubits
):
    controlled_gate = ControlledGate(target_gate, control=control)
    assert controlled_gate.qubits == expected_qubits


@pytest.mark.parametrize(
    "controlled_gate, expected_matrix",
    [
        (ControlledGate(X(0), 1), sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
        (
            ControlledGate(
                CustomGate(
                    sympy.Matrix([
                        [sympy.cos(THETA), -sympy.sin(THETA)],
                        [sympy.sin(THETA), sympy.cos(THETA)]
                    ]),
                    (2,)
                ),
                1
            ),
            sympy.Matrix([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, sympy.cos(THETA), -sympy.sin(THETA)],
                [0, 0, sympy.sin(THETA), sympy.cos(THETA)]
            ])
        ),
        (
            ControlledGate(ControlledGate(X(3), 2), 1),
            sympy.Matrix([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0]
            ])
        )
    ]
)
def test_matrix_of_controlled_gate_comprises_identity_and_target_matrix_block(
    controlled_gate, expected_matrix
):
    assert controlled_gate.matrix == expected_matrix
