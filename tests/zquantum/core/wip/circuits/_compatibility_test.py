import numpy as np
import pyquil
import pyquil.gates
import pytest
import sympy
import zquantum.core.circuit as old_circuit
import zquantum.core.wip.circuits as new_circuits
from zquantum.core.wip.circuits._compatibility import (
    new_circuit_from_old_circuit,
    old_circuit_from_new_circuit,
)

PYQUIL_PROGRAMS = [
    pyquil.Program(),
    pyquil.Program(pyquil.gates.X(2), pyquil.gates.Y(0)),
    pyquil.Program(pyquil.gates.CNOT(3, 1)),
    pyquil.Program(pyquil.gates.RX(np.pi, 1)),
]


def _old_circuit_from_pyquil(program):
    return old_circuit.Circuit(program)


def _new_circuit_from_pyquil(program):
    return new_circuits.import_from_pyquil(program)


THETA_1 = sympy.Symbol("theta_1")


def _make_old_parametric_circuit():
    qubit = old_circuit.Qubit(0)
    gate_RX = old_circuit.Gate("Rx", params=[THETA_1], qubits=[qubit])

    circ = old_circuit.Circuit()
    circ.qubits = [qubit]
    circ.gates = [gate_RX]

    return circ


EQUIVALENT_OLD_NEW_CIRCUITS = [
    *[
        (_old_circuit_from_pyquil(program), _new_circuit_from_pyquil(program))
        for program in PYQUIL_PROGRAMS
    ],
    (
        _make_old_parametric_circuit(),
        new_circuits.Circuit([new_circuits.RX(THETA_1)(0)]),
    ),
]


@pytest.mark.parametrize(
    "old,new",
    EQUIVALENT_OLD_NEW_CIRCUITS,
)
def test_translating_old_circuit_matches_new_circuit(old, new):
    assert new_circuit_from_old_circuit(old) == new


@pytest.mark.parametrize(
    "old,new",
    EQUIVALENT_OLD_NEW_CIRCUITS,
)
def test_translating_new_circuit_matches_old_circuit(old, new):
    assert old_circuit_from_new_circuit(new) == old
