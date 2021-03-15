import numpy as np
import pytest
import pyquil

from .pyquil_conversions import export_to_pyquil, import_from_pyquil
from .. import _gates
from .. import _builtin_gates


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
    )
]


class TestExportingToPyQuil:
    @pytest.mark.parametrize("zquantum_circuit, pyquil_circuit", EQUIVALENT_CIRCUITS)
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, pyquil_circuit
    ):
        exported = export_to_pyquil(zquantum_circuit)
        print(exported.instructions, pyquil_circuit.instructions)
        assert exported == pyquil_circuit


class TestImportingFromPyQuil:
    @pytest.mark.parametrize("zquantum_circuit, pyquil_circuit", EQUIVALENT_CIRCUITS)
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, pyquil_circuit
    ):
        imported = import_from_pyquil(pyquil_circuit)
        assert imported == zquantum_circuit
