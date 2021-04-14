import pytest
import pyquil
import pyquil.gates

import zquantum.core.circuit as old_circuit
from zquantum.core.wip.circuits._compat import new_circuit_from_old_circuit
import zquantum.core.wip.circuits as new_circuits


PYQUIL_PROGRAMS = [
    pyquil.Program([]),
    pyquil.Program([pyquil.gates.X(2), pyquil.gates.Y(0)]),
]


def _old_circuit_from_pyquil(program):
    return old_circuit.Circuit(program)


def _new_circuit_from_pyquil(program):
    return new_circuits.import_from_pyquil(program)


@pytest.mark.parametrize(
    "old,new",
    [
        (_old_circuit_from_pyquil(program), _new_circuit_from_pyquil(program))
        for program in PYQUIL_PROGRAMS
    ],
)
def test_translated_circuit_matches_expected_circuit(old, new):
    assert new_circuit_from_old_circuit(old) == new
