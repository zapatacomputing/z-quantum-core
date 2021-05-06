import numpy as np
import pyquil
import pyquil.gates
import pytest
import sympy
import zquantum.core.circuit as old_circuit
import zquantum.core.wip.circuits as new_circuits
from zquantum.core.wip.circuits._compatibility import new_circuit_from_old_circuit

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


def _make_old_circuit_with_inactive_qubits(x_qubit, cnot_qubits, n_qubits):
    circuit = old_circuit.Circuit(
        pyquil.Program(pyquil.gates.X(x_qubit), pyquil.gates.CNOT(*cnot_qubits))
    )
    circuit.qubits = [old_circuit.Qubit(i) for i in range(n_qubits)]
    return circuit


@pytest.mark.parametrize(
    "old,new",
    [
        *[
            (_old_circuit_from_pyquil(program), _new_circuit_from_pyquil(program))
            for program in PYQUIL_PROGRAMS
        ],
        (
            _make_old_parametric_circuit(),
            new_circuits.Circuit([new_circuits.RX(THETA_1)(0)]),
        ),
        *[
            (
                _make_old_circuit_with_inactive_qubits(x_qubit, cnot_qubits, n_qubits),
                new_circuits.Circuit(
                    [new_circuits.X(x_qubit), new_circuits.CNOT(*cnot_qubits)],
                    n_qubits=n_qubits,
                ),
            )
            for x_qubit, cnot_qubits, n_qubits in [
                (0, (1, 2), 4),
                (1, (3, 4), 5),
                (0, (2, 3), 4),
            ]
        ],
    ],
)
def test_translated_circuit_matches_expected_circuit(old, new):
    assert new_circuit_from_old_circuit(old) == new
