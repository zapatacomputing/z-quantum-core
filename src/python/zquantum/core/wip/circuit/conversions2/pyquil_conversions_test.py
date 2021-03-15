import numpy as np
import pytest
import pyquil
import sympy

from .pyquil_conversions import export_to_pyquil, import_from_pyquil
from .. import _gates
from .. import _builtin_gates


SYMPY_THETA = sympy.Symbol("theta")
SYMPY_GAMMA = sympy.Symbol("gamma")
QUIL_THETA = pyquil.quil.Parameter("theta")
QUIL_GAMMA = pyquil.quil.Parameter("gamma")

EQUIVALENT_CIRCUITS = [
    (
        _gates.Circuit([], 0),
        pyquil.Program([]),
    ),
    (
        _gates.Circuit([
            _builtin_gates.X(2),
            _builtin_gates.Y(0),
        ]),
        pyquil.Program([
            pyquil.gates.X(2),
            pyquil.gates.Y(0)
        ]),
    ),
    (
        _gates.Circuit([
            _builtin_gates.CNOT(3, 1),
        ]),
        pyquil.Program([
            pyquil.gates.CNOT(3, 1),
        ]),
    ),
    (
        _gates.Circuit([
            _builtin_gates.RX(np.pi)(1)
        ]),
        pyquil.Program([
            pyquil.gates.RX(np.pi, 1)
        ])
    ),
    (
        _gates.Circuit(
            [_builtin_gates.SWAP.controlled(1)(2, 0, 3)],
        ),
        pyquil.Program([
            pyquil.gates.SWAP(0, 3).controlled(2)
        ])
    ),
    (
        _gates.Circuit(
            [_builtin_gates.Y.controlled(2)(3, 1, 2)]
        ),
        pyquil.Program([
            pyquil.gates.Y(2).controlled(1).controlled(3)
        ])
    ),
    (
        _gates.Circuit(
            [_builtin_gates.RX(0.5).dagger.controlled(2)(3, 1, 2)]
        ),
        pyquil.Program([
            pyquil.gates.RX(0.5, 2).dagger().controlled(1).controlled(3)
        ])
    ),
    (
        _gates.Circuit(
            [_builtin_gates.RX(0.5).controlled(2).dagger(3, 1, 2)]
        ),
        pyquil.Program([
            pyquil.gates.RX(0.5, 2).dagger().controlled(1).controlled(3)
        ])
    )
]


EQUIVALENT_PARAMETRIZED_CIRCUITS = [
    (
        _gates.Circuit(
            [
                _builtin_gates.RX(SYMPY_THETA)(1),
            ],
        ),
        pyquil.Program([
            pyquil.quil.Declare("theta", "REAL"),
            pyquil.gates.RX(QUIL_THETA, 1)
        ])
    ),
    (
        _gates.Circuit(
            [
                _builtin_gates.RX(sympy.Mul(SYMPY_GAMMA, SYMPY_THETA, evaluate=False))(1),
            ],
        ),
        pyquil.Program([
            pyquil.quil.Declare("gamma", "REAL"),
            pyquil.quil.Declare("theta", "REAL"),
            pyquil.gates.RX(QUIL_GAMMA * QUIL_THETA, 1)
        ])
    ),
]


class TestExportingToPyQuil:
    @pytest.mark.parametrize(
        "zquantum_circuit, pyquil_circuit",
        [*EQUIVALENT_CIRCUITS, *EQUIVALENT_PARAMETRIZED_CIRCUITS])
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, pyquil_circuit
    ):
        exported = export_to_pyquil(zquantum_circuit)
        assert exported == pyquil_circuit, (exported.instructions, pyquil_circuit.instructions)


class TestImportingFromPyQuil:
    @pytest.mark.parametrize(
        "zquantum_circuit, pyquil_circuit",
        [*EQUIVALENT_CIRCUITS, *EQUIVALENT_PARAMETRIZED_CIRCUITS])
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, pyquil_circuit
    ):
        imported = import_from_pyquil(pyquil_circuit)
        assert imported == zquantum_circuit
