from .single_qubit_gates import X, Y, Z, H, I, Phase, T, Rx, Ry, Rz
from ._gate import Gate
import pytest
import numpy as np
import sympy


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_creating_X_gate(qubit):
    gate = X(qubit)
    Xgate = Gate(sympy.Matrix([[0, 1], [1, 0]]), (qubit,))
    assert gate == Xgate


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_creating_Y_gate(qubit):
    gate = Y(qubit)
    Ygate = Gate(
        sympy.Matrix([[complex(0, 0), complex(0, -1)], [complex(0, 1), complex(0, 0)]]),
        (qubit,),
    )
    assert gate == Ygate


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_creating_Z_gate(qubit):
    gate = Z(qubit)
    Zgate = Gate(
        sympy.Matrix([[1, 0], [0, -1]]),
        (qubit,),
    )
    assert gate == Zgate


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_creating_H_gate(qubit):
    gate = H(qubit)
    Hgate = Gate(
        sympy.Matrix(
            [
                [(1 / np.sqrt(2)), (1 / np.sqrt(2))],
                [(1 / np.sqrt(2)), -1 * (1 / np.sqrt(2))],
            ]
        ),
        (qubit,),
    )
    assert gate == Hgate


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_creating_I_gate(qubit):
    gate = I(qubit)
    Igate = Gate(
        sympy.Matrix(
            [
                [1, 0],
                [0, 1],
            ]
        ),
        (qubit,),
    )
    assert gate == Igate


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_creating_Phase_gate(qubit):
    gate = Phase(qubit)
    Phasegate = Gate(
        sympy.Matrix(
            [
                [1, 0],
                [0, complex(0, 1)],
            ]
        ),
        (qubit,),
    )
    assert gate == Phasegate


@pytest.mark.parametrize(
    "qubit",
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 99, 100, 101],
)
def test_creating_T_gate(qubit):
    gate = T(qubit)
    Tgate = Gate(
        sympy.Matrix(
            [
                [1, 0],
                [0, sympy.exp(complex(0, np.pi / 4))],
            ]
        ),
        (qubit,),
    )
    assert gate == Tgate


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
def test_creating_Rx_gate(qubit, theta):
    if theta is not None:
        gate = Rx(qubit, theta)
        Rxgate = Gate(
            sympy.Matrix(
                [
                    [
                        sympy.cos(theta / 2),
                        -sympy.I * sympy.sin(theta / 2),
                    ],
                    [
                        -sympy.I * sympy.sin(theta / 2),
                        sympy.cos(theta / 2),
                    ],
                ]
            ),
            (qubit,),
        )
    else:
        gate = Rx(qubit)
        Rxgate = Gate(
            sympy.Matrix(
                [
                    [
                        sympy.cos(sympy.Symbol("theta") / 2),
                        -sympy.I * sympy.sin(sympy.Symbol("theta") / 2),
                    ],
                    [
                        -sympy.I * sympy.sin(sympy.Symbol("theta") / 2),
                        sympy.cos(sympy.Symbol("theta") / 2),
                    ],
                ]
            ),
            (qubit,),
        )
    assert gate == Rxgate


@pytest.mark.parametrize("qubit, theta", rotation_gate_test_data)
def test_creating_Ry_gate(qubit, theta):
    if theta is not None:
        gate = Ry(qubit, theta)
        Rygate = Gate(
            sympy.Matrix(
                [
                    [
                        sympy.cos(theta / 2),
                        -1 * sympy.sin(theta / 2),
                    ],
                    [
                        sympy.sin(theta / 2),
                        sympy.cos(theta / 2),
                    ],
                ]
            ),
            (qubit,),
        )
    else:
        gate = Ry(qubit)
        Rygate = Gate(
            sympy.Matrix(
                [
                    [
                        sympy.cos(sympy.Symbol("theta") / 2),
                        -1 * sympy.sin(sympy.Symbol("theta") / 2),
                    ],
                    [
                        sympy.sin(sympy.Symbol("theta") / 2),
                        sympy.cos(sympy.Symbol("theta") / 2),
                    ],
                ]
            ),
            (qubit,),
        )
    assert gate == Rygate


@pytest.mark.parametrize("qubit, theta", rotation_gate_test_data)
def test_creating_Rz_gate(qubit, theta):
    if theta is not None:
        gate = Rz(qubit, theta)
        Rzgate = Gate(
            sympy.Matrix(
                [
                    [sympy.exp(-1 * sympy.I * theta / 2), 0],
                    [0, sympy.exp(sympy.I * theta / 2)],
                ]
            ),
            (qubit,),
        )
    else:
        gate = Rz(qubit)
        Rzgate = Gate(
            sympy.Matrix(
                [
                    [sympy.exp(-1 * sympy.I * sympy.Symbol("theta") / 2), 0],
                    [0, sympy.exp(sympy.I * sympy.Symbol("theta") / 2)],
                ]
            ),
            (qubit,),
        )
    assert gate == Rzgate