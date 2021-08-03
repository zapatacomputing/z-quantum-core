import numpy as np
import pyquil
import pytest
import sympy
from zquantum.core.circuits import _builtin_gates, _circuit, _gates
from zquantum.core.circuits.conversions.pyquil_conversions import (
    export_to_pyquil,
    import_from_pyquil,
)

SYMPY_GAMMA = sympy.Symbol("gamma")
QUIL_GAMMA = pyquil.quil.Parameter("gamma")

"""
Note: Those differently named Symbols/Parameters are needed due
to the manner the conversion logic abstracts the names of the Symbols

Example:
SYMPY_THETA = sympy.Symbol("theta_0")
QUIL_THETA = pyquil.quil.Parameter("theta")

SYMPY_THETA = sympy.Symbol("theta")
QUIL_THETA = pyquil.quil.Parameter("theta")
----------------------------------------------
Export fails, Import passes
"""
SYMPY_THETA_0 = sympy.Symbol("theta_0")
SYMPY_THETA_1 = sympy.Symbol("theta_1")
SYMPY_THETA_2 = sympy.Symbol("theta_2")
QUIL_THETA_0 = pyquil.quil.Parameter("theta_0")
QUIL_THETA_1 = pyquil.quil.Parameter("theta_1")
QUIL_THETA_2 = pyquil.quil.Parameter("theta_2")


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

PYQUIL_XX = pyquil.quil.DefGate(
    name="XX",
    matrix=[
        [
            pyquil.quilatom.quil_cos(0.5 * QUIL_THETA_0),
            0,
            0,
            -1j * pyquil.quilatom.quil_sin(0.5 * QUIL_THETA_0),
        ],
        [
            0,
            pyquil.quilatom.quil_cos(0.5 * QUIL_THETA_0),
            -1j * pyquil.quilatom.quil_sin(0.5 * QUIL_THETA_0),
            0,
        ],
        [
            0,
            -1j * pyquil.quilatom.quil_sin(0.5 * QUIL_THETA_0),
            pyquil.quilatom.quil_cos(0.5 * QUIL_THETA_0),
            0,
        ],
        [
            -1j * pyquil.quilatom.quil_sin(0.5 * QUIL_THETA_0),
            0,
            0,
            pyquil.quilatom.quil_cos(0.5 * QUIL_THETA_0),
        ],
    ],
    parameters=[QUIL_THETA_0],
)


def pyquil_rh_definition():
    cos_term = pyquil.quilatom.quil_cos(0.5 * QUIL_THETA_0)
    sin_term = pyquil.quilatom.quil_sin(0.5 * QUIL_THETA_0)
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
        parameters=[QUIL_THETA_0],
    )


def pyquil_u3_definition():
    # Note: need to add an extra global phase to match to z-quantum's definition
    cos_term = pyquil.quilatom.quil_cos(0.5 * QUIL_THETA_0)
    sin_term = pyquil.quilatom.quil_sin(0.5 * QUIL_THETA_0)

    global_phase_phi = pyquil.quilatom.quil_exp(0.5j * QUIL_THETA_1)
    global_phase_lambda = pyquil.quilatom.quil_exp(0.5j * QUIL_THETA_2)
    global_phase_phi_neg = pyquil.quilatom.quil_exp(-0.5j * QUIL_THETA_1)
    global_phase_lambda_neg = pyquil.quilatom.quil_exp(-0.5j * QUIL_THETA_2)

    return pyquil.quil.DefGate(
        name="U3",
        matrix=[
            [
                cos_term * global_phase_phi_neg * global_phase_lambda_neg,
                -1 * global_phase_lambda * global_phase_phi_neg * sin_term,
            ],
            [
                global_phase_phi * global_phase_lambda_neg * sin_term,
                cos_term * global_phase_phi * global_phase_lambda,
            ],
        ],
        parameters=[QUIL_THETA_0, QUIL_THETA_1, QUIL_THETA_2],
    )


PYQUIL_RH = pyquil_rh_definition()
PYQUIL_U3 = pyquil_u3_definition()


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
    (
        _circuit.Circuit([_builtin_gates.U3(np.pi / 2, np.pi / 4, 0)(3)]),
        pyquil.Program(
            [PYQUIL_U3, PYQUIL_U3.get_constructor()(np.pi / 2, np.pi / 4, 0)(3)]
        ),
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
        pyquil.quil.Declare(QUIL_THETA_0.name, "REAL"),
        gate_def,
        gate_constructor(QUIL_THETA_0)(0),
    )


def _example_rh_pyquil_program():
    gate_def = PYQUIL_RH
    gate_constructor = gate_def.get_constructor()

    return pyquil.Program(
        pyquil.quil.Declare(QUIL_THETA_0.name, "REAL"),
        gate_def,
        gate_constructor(QUIL_THETA_0)(0),
    )


def _example_xx_pyquil_program():
    gate_def = PYQUIL_XX
    gate_constructor = gate_def.get_constructor()

    return pyquil.Program(
        pyquil.quil.Declare(QUIL_THETA_0.name, "REAL"),
        gate_def,
        gate_constructor(QUIL_THETA_0)(0),
    )


def _example_u3_pyquil_program():
    gate_def = PYQUIL_U3
    gate_constructor = gate_def.get_constructor()

    return pyquil.Program(
        pyquil.quil.Declare(QUIL_THETA_0.name, "REAL"),
        pyquil.quil.Declare(QUIL_THETA_1.name, "REAL"),
        pyquil.quil.Declare(QUIL_THETA_2.name, "REAL"),
        gate_def,
        gate_constructor(QUIL_THETA_0, QUIL_THETA_1, QUIL_THETA_2)(0),
    )


EQUIVALENT_PARAMETRIZED_CIRCUITS = [
    (
        _circuit.Circuit(
            [
                _builtin_gates.RX(SYMPY_THETA_0)(1),
            ],
        ),
        pyquil.Program(
            [
                pyquil.quil.Declare(QUIL_THETA_0.name, "REAL"),
                pyquil.gates.RX(QUIL_THETA_0, 1),
            ]
        ),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.RX(
                    sympy.Mul(SYMPY_THETA_0, SYMPY_THETA_1, evaluate=False)
                )(1),
            ],
        ),
        pyquil.Program(
            [
                pyquil.quil.Declare(QUIL_THETA_0.name, "REAL"),
                pyquil.quil.Declare(QUIL_THETA_1.name, "REAL"),
                pyquil.gates.RX(QUIL_THETA_0 * QUIL_THETA_1, 1),
            ]
        ),
    ),
    (
        _circuit.Circuit(
            [
                CUSTOM_PARAMETRIC_DEF(SYMPY_THETA_0)(0),
            ],
        ),
        _example_parametric_pyquil_program(),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.RH(SYMPY_THETA_0)(0),
            ],
        ),
        _example_rh_pyquil_program(),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.XX(SYMPY_THETA_0)(0),
            ],
        ),
        _example_xx_pyquil_program(),
    ),
    (
        _circuit.Circuit(
            [
                _builtin_gates.U3(SYMPY_THETA_0, SYMPY_THETA_1, SYMPY_THETA_2)(0),
            ],
        ),
        _example_u3_pyquil_program(),
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
