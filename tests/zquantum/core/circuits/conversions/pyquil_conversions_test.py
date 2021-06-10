import numpy as np
import pyquil
import pytest
import sympy
from zquantum.core.circuits import _builtin_gates, _circuit, _gates
from zquantum.core.circuits.conversions.pyquil_conversions import (
    export_to_pyquil,
    import_from_pyquil,
)

SYMPY_THETA = sympy.Symbol("theta")
SYMPY_GAMMA = sympy.Symbol("gamma")
QUIL_THETA = pyquil.quil.Parameter("theta")
QUIL_GAMMA = pyquil.quil.Parameter("gamma")

SQRT_X_DEF = _gates.CustomGateDefinition(
    "SQRT-X",
    sympy.Matrix([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]),
    tuple(),
)
CUSTOM_PARAMETRIC_DEF = _gates.CustomGateDefinition(
    "CUSTOM-PARAMETRIC",
    sympy.Matrix(
        [
            [sympy.cos(SYMPY_GAMMA), sympy.sin(SYMPY_GAMMA)],
            [-sympy.sin(SYMPY_GAMMA), sympy.cos(SYMPY_GAMMA)],
        ]
    ),
    (SYMPY_GAMMA,),
)

PYQUIL_THETA_0 = pyquil.quil.Parameter("theta_0")

PYQUIL_XX = pyquil.quil.DefGate(
    name="XX",
    matrix=[
        [
            pyquil.quilatom.quil_cos(0.5 * PYQUIL_THETA_0),
            0,
            0,
            -1j * pyquil.quilatom.quil_sin(0.5 * PYQUIL_THETA_0),
        ],
        [
            0,
            pyquil.quilatom.quil_cos(0.5 * PYQUIL_THETA_0),
            -1j * pyquil.quilatom.quil_sin(0.5 * PYQUIL_THETA_0),
            0,
        ],
        [
            0,
            -1j * pyquil.quilatom.quil_sin(0.5 * PYQUIL_THETA_0),
            pyquil.quilatom.quil_cos(0.5 * PYQUIL_THETA_0),
            0,
        ],
        [
            -1j * pyquil.quilatom.quil_sin(0.5 * PYQUIL_THETA_0),
            0,
            0,
            pyquil.quilatom.quil_cos(0.5 * PYQUIL_THETA_0),
        ],
    ],
    parameters=[PYQUIL_THETA_0],
)


def pyquil_rh_definition():
    cos_term = pyquil.quilatom.quil_cos(0.5 * PYQUIL_THETA_0)
    sin_term = pyquil.quilatom.quil_sin(0.5 * PYQUIL_THETA_0)
    phase_factor = 1j * sin_term + cos_term
    return pyquil.quil.DefGate(
        name="RH",
        matrix=[
            [
                phase_factor
                * (-0.5j * pyquil.quilatom.quil_sqrt(2) * sin_term + cos_term),
                -0.5j * pyquil.quilatom.quil_sqrt(2) * phase_factor * sin_term,
            ],
            [
                -0.5j * pyquil.quilatom.quil_sqrt(2) * phase_factor * sin_term,
                phase_factor
                * (0.5j * pyquil.quilatom.quil_sqrt(2) * sin_term + cos_term),
            ],
        ],
        parameters=[PYQUIL_THETA_0],
    )


PYQUIL_RH = pyquil_rh_definition()


EQUIVALENT_CIRCUITS = [
    (
        _circuit.Circuit([], 0),
        pyquil.Program([]),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.X(2),
                _builtin_gates.Y(0),
            ]
        ),
        pyquil.Program([pyquil.gates.X(2), pyquil.gates.Y(0)]),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.CNOT(3, 1),
            ]
        ),
        pyquil.Program(
            [
                pyquil.gates.CNOT(3, 1),
            ]
        ),
    ),
    (
        _circuit.Circuit([_builtin_gates.RX(np.pi)(1)]),
        pyquil.Program([pyquil.gates.RX(np.pi, 1)]),
    ),
    (
        _circuit.Circuit([_builtin_gates.S(1)]),
        pyquil.Program([pyquil.gates.S(1)]),
    ),
    (
        _circuit.Circuit(
            [_builtin_gates.SWAP.controlled(1)(2, 0, 3)],
        ),
        pyquil.Program([pyquil.gates.SWAP(0, 3).controlled(2)]),
    ),
    (
        _circuit.Circuit([_builtin_gates.Y.controlled(2)(3, 1, 2)]),
        pyquil.Program([pyquil.gates.Y(2).controlled(1).controlled(3)]),
    ),
    (
        _circuit.Circuit([_builtin_gates.RX(0.5).dagger.controlled(2)(3, 1, 2)]),
        pyquil.Program([pyquil.gates.RX(0.5, 2).dagger().controlled(1).controlled(3)]),
    ),
    (
        _circuit.Circuit([_builtin_gates.RX(0.5).controlled(2).dagger(3, 1, 2)]),
        pyquil.Program([pyquil.gates.RX(0.5, 2).dagger().controlled(1).controlled(3)]),
    ),
    (
        _circuit.Circuit(
            [SQRT_X_DEF()(3)],
        ),
        pyquil.Program([("SQRT-X", 3)]).defgate(
            "SQRT-X",
            np.array(
                [
                    [0.5 + 0.5j, 0.5 - 0.5j],
                    [0.5 - 0.5j, 0.5 + 0.5j],
                ]
            ),
        ),
    ),
    (
        _circuit.Circuit([_builtin_gates.XX(np.pi)(2)]),
        pyquil.Program([PYQUIL_XX, PYQUIL_XX.get_constructor()(np.pi)(2)]),
    ),
    (
        _circuit.Circuit([_builtin_gates.RH(np.pi / 5)(3)]),
        pyquil.Program([PYQUIL_RH, PYQUIL_RH.get_constructor()(np.pi / 5)(3)]),
    ),
]


def _example_parametric_pyquil_program():
    gate_def = pyquil.quil.DefGate(
        "CUSTOM-PARAMETRIC",
        [
            [
                pyquil.quilatom.quil_cos(QUIL_GAMMA),
                pyquil.quilatom.quil_sin(QUIL_GAMMA),
            ],
            [
                -pyquil.quilatom.quil_sin(QUIL_GAMMA),
                pyquil.quilatom.quil_cos(QUIL_GAMMA),
            ],
        ],
        [QUIL_GAMMA],
    )
    gate_constructor = gate_def.get_constructor()

    return pyquil.Program(
        pyquil.quil.Declare("theta", "REAL"),
        gate_def,
        gate_constructor(QUIL_THETA)(0),
    )


EQUIVALENT_PARAMETRIZED_CIRCUITS = [
    (
        _circuit.Circuit(
            [
                _builtin_gates.RX(SYMPY_THETA)(1),
            ],
        ),
        pyquil.Program(
            [pyquil.quil.Declare("theta", "REAL"), pyquil.gates.RX(QUIL_THETA, 1)]
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.RX(sympy.Mul(SYMPY_GAMMA, SYMPY_THETA, evaluate=False))(
                    1
                ),
            ],
        ),
        pyquil.Program(
            [
                pyquil.quil.Declare("gamma", "REAL"),
                pyquil.quil.Declare("theta", "REAL"),
                pyquil.gates.RX(QUIL_GAMMA * QUIL_THETA, 1),
            ]
        ),
    ),
    (
        _circuit.Circuit(
            [
                CUSTOM_PARAMETRIC_DEF(SYMPY_THETA)(0),
            ],
        ),
        _example_parametric_pyquil_program(),
    ),
]


class TestExportingToPyQuil:
    @pytest.mark.parametrize(
        "zquantum_circuit, pyquil_circuit",
        [*EQUIVALENT_CIRCUITS, *EQUIVALENT_PARAMETRIZED_CIRCUITS],
    )
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, pyquil_circuit
    ):
        exported = export_to_pyquil(zquantum_circuit)
        assert exported == pyquil_circuit, (
            exported.out(),
            pyquil_circuit.out(),
        )


class TestImportingFromPyQuil:
    @pytest.mark.parametrize(
        "zquantum_circuit, pyquil_circuit",
        [*EQUIVALENT_CIRCUITS, *EQUIVALENT_PARAMETRIZED_CIRCUITS],
    )
    def test_importing_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, pyquil_circuit
    ):
        imported = import_from_pyquil(pyquil_circuit)
        assert imported == zquantum_circuit
