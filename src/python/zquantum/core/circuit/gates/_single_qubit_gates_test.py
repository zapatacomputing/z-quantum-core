from . import X, Y, Z, H, I, PHASE, T, RX, RY, RZ
from . import CustomGate
import numpy as np
import pytest
import sympy


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_X_gate_has_correct_matrix(qubit):
    assert X(qubit) == CustomGate(sympy.Matrix([[0, 1], [1, 0]]), (qubit,))


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_Y_gate_has_correct_matrix(qubit):
    assert Y(qubit) == CustomGate(sympy.Matrix([[0, -1.0j], [1j, 0]]), (qubit,))


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_Z_gate_has_correct_matrix(qubit):
    assert Z(qubit) == CustomGate(sympy.Matrix([[1, 0], [0, -1]]), (qubit,))


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_hadamard_gate_has_correct_matrix(qubit):
    assert H(qubit) == CustomGate(
        sympy.Matrix(
            [
                [(1 / np.sqrt(2)), (1 / np.sqrt(2))],
                [(1 / np.sqrt(2)), -1 * (1 / np.sqrt(2))],
            ]
        ),
        (qubit,),
    )


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_identity_gate_has_correct_matrix(qubit):
    assert I(qubit) == CustomGate(sympy.Matrix([[1, 0], [0, 1]]), (qubit,))


@pytest.mark.parametrize("angle", [np.pi, np.pi / 2, 0.1, 0])
@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_PHASE_gate_has_correct_matrix(qubit, angle):
    assert PHASE(qubit, angle) == CustomGate(
        sympy.Matrix([[1, 0], [0, sympy.exp(1j * angle)]]), (qubit,)
    )


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_T_gate_has_correct_matrix(qubit):
    assert T(qubit) == CustomGate(
        sympy.Matrix([[1, 0], [0, sympy.exp(1j * np.pi / 4)]]), (qubit,)
    )


rotation_gate_test_data = [
    [qubit, theta]
    for qubit in [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101]
    for theta in [
        -2 * np.pi,
        -np.pi,
        -np.pi / 2,
        0,
        1,
        np.pi / 2,
        np.pi,
        2 * np.pi,
        1.012,
        -0.0001,
        sympy.Symbol("theta"),
        sympy.Symbol("beta"),
        sympy.Symbol("Gamma"),
        None,
    ]
]


@pytest.mark.parametrize("qubit, theta", rotation_gate_test_data)
class TestCreatingRotationGates:
    def test_RX_gate_has_correct_matrix(self, qubit, theta):
        theta = theta if theta is not None else sympy.Symbol("theta")
        assert RX(qubit, theta) == CustomGate(
            sympy.Matrix(
                [
                    [sympy.cos(theta / 2), -sympy.I * sympy.sin(theta / 2)],
                    [-sympy.I * sympy.sin(theta / 2), sympy.cos(theta / 2)],
                ]
            ),
            (qubit,),
        )

    def test_RY_gate_has_correct_matrix(self, qubit, theta):
        theta = theta if theta is not None else sympy.Symbol("theta")
        assert RY(qubit, theta) == CustomGate(
            sympy.Matrix(
                [
                    [sympy.cos(theta / 2), -1 * sympy.sin(theta / 2)],
                    [sympy.sin(theta / 2), sympy.cos(theta / 2)],
                ]
            ),
            (qubit,),
        )

    def test_RZ_gate_has_correct_matrix(self, qubit, theta):
        theta = theta if theta is not None else sympy.Symbol("theta")
        assert RZ(qubit, theta) == CustomGate(
            sympy.Matrix(
                [
                    [sympy.exp(-1 * sympy.I * theta / 2), 0],
                    [0, sympy.exp(sympy.I * theta / 2)],
                ]
            ),
            (qubit,),
        )


@pytest.mark.parametrize("rotation_gate", [RX, RY, RZ])
def test_rotation_gate_with_parameter_equal_to_zero_is_equivalent_to_identity_matrix(
    rotation_gate,
):
    assert rotation_gate(0, 0) == I(0)


@pytest.mark.parametrize(
    "theta, evaluated_matrix",
    [
        (
            np.pi,
            sympy.Matrix(
                [
                    [0, complex(0, -1)],
                    [complex(0, -1), 0],
                ]
            ),
        ),
        (
            2 * np.pi,
            sympy.Matrix(
                [
                    [-1, 0],
                    [0, -1],
                ]
            ),
        ),
        (
            np.pi / 2,
            sympy.Matrix(
                [
                    [0.7071067811865476, complex(0, -0.7071067811865475)],
                    [complex(0, -0.7071067811865475), 0.7071067811865476],
                ]
            ),
        ),
    ],
)
def test_evaluating_rx_gate_results_in_matrix_with_correct_entries(
    theta, evaluated_matrix
):
    qubits = (0,)
    matrix = sympy.Matrix(
        [
            [
                sympy.cos(sympy.Symbol("theta") / 2),
                -1j * sympy.sin(sympy.Symbol("theta") / 2),
            ],
            [
                -1j * sympy.sin(sympy.Symbol("theta") / 2),
                sympy.cos(sympy.Symbol("theta") / 2),
            ],
        ]
    )
    gate = CustomGate(matrix, qubits)
    symbols_map = {"theta": theta}
    expected_new_gate = CustomGate(evaluated_matrix, qubits)

    new_gate = gate.evaluate(symbols_map)

    assert new_gate == expected_new_gate


@pytest.mark.parametrize("gate_cls", [H, T, I, X, Y, Z])
def test_nonparametric_single_qubit_gates_have_no_params(gate_cls):
    assert gate_cls(1).params == ()


@pytest.mark.parametrize("gate_cls", [RX, RY, RZ])
@pytest.mark.parametrize("angle", [sympy.Symbol("alpha"), np.pi / 2])
def test_rotation_gates_have_a_single_parameter_equal_to_their_angle(gate_cls, angle):
    assert gate_cls(1, angle).params == (angle,)


@pytest.mark.parametrize(
    "gate",
    [
        H(0),
        T(1),
        I(2),
        X(5),
        Y(1),
        Z(2),
        RX(0, np.pi / 2),
        RY(0, sympy.Symbol("alpha")),
        RZ(1, sympy.Symbol("alpha") + sympy.Symbol("x")),
    ],
)
@pytest.mark.parametrize(
    "symbol_map",
    [
        {},
        {sympy.Symbol("beta"): 0.5 * np.pi},
        {sympy.Symbol("alpha"): sympy.Symbol("gamma")},
        {sympy.Symbol("alpha"): sympy.Symbol("x") + sympy.Symbol("y")},
    ],
)
class TestEvaluationOfSingleQubitGates:
    def test_evaluating_single_qubit_gate_preserves_gate_type(self, gate, symbol_map):
        assert type(gate.evaluate(symbol_map)) == type(gate)

    def test_evaluating_single_qubit_gate_preserves_qubits(self, gate, symbol_map):
        assert gate.evaluate(symbol_map).qubits == gate.qubits

    def test_evaluating_single_qubit_gate_correctly_substitutes_parameters(
        self, gate, symbol_map
    ):
        expected_params = tuple(
            param.subs(symbol_map) if isinstance(param, sympy.Basic) else param
            for param in gate.params
        )

        assert gate.evaluate(symbol_map).params == expected_params
