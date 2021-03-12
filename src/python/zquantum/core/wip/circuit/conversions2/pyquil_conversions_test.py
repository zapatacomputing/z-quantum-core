import pytest
import pyquil

from .pyquil_conversions import export_to_pyquil, import_from_pyquil
from .. import _gates


EQUIVALENT_CIRCUITS = [
    (
        _gates.Circuit([], 0),
        pyquil.Program([]),
    ),
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
