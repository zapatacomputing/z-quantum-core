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
    )
]


class TestExportingToPyQuil:
    @pytest.mark.parametrize("zquantum_circuit, pyquil_circuit", EQUIVALENT_CIRCUITS)
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, pyquil_circuit
    ):
        exported = export_to_pyquil(zquantum_circuit)
        assert exported == pyquil_circuit


class TestImportingFromPyQuil:
    @pytest.mark.parametrize("zquantum_circuit, pyquil_circuit", EQUIVALENT_CIRCUITS)
    def test_exporting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, pyquil_circuit
    ):
        imported = import_from_pyquil(pyquil_circuit)
        assert imported == zquantum_circuit