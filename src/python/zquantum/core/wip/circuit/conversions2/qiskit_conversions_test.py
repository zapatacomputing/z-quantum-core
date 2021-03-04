import sympy
import numpy as np
import qiskit
import pytest

from .qiskit_conversions import convert_to_qiskit
from .. import _gates as g
from .. import _builtin_gates as bg


# NOTE: In Qiskit, 0 is the most significant qubit,
# whereas in ZQuantum, 0 is the least significant qubit.
# This is we need to flip the indices.
#
# See more at
# https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html#Visualize-Circuit


def _single_qubit_qiskit_circuit():
    qc = qiskit.QuantumCircuit(6)
    qc.x(0)
    qc.z(2)
    return qc


def _two_qubit_qiskit_circuit():
    qc = qiskit.QuantumCircuit(4)
    qc.cnot(0, 1)
    return qc


def _parametric_qiskit_circuit(angle):
    qc = qiskit.QuantumCircuit(4)
    qc.rx(angle, 1)
    return qc


def _qiskit_circuit_with_controlled_gate():
    qc = qiskit.QuantumCircuit(5)
    qc.append(qiskit.circuit.library.SwapGate().control(1), [2, 0, 3])
    return qc


def _qiskit_circuit_with_multicontrolled_gate():
    qc = qiskit.QuantumCircuit(6)
    qc.append(qiskit.circuit.library.YGate().control(2), [4, 5, 2])
    return qc


SYMPY_THETA = sympy.Symbol("theta")
SYMPY_GAMMA = sympy.Symbol("gamma")
QISKIT_THETA = qiskit.circuit.Parameter("theta")
QISKIT_GAMMA = qiskit.circuit.Parameter("gamma")


EQUIVALENT_CIRCUITS = [
    (
        g.Circuit(
            [
                bg.X(0),
                bg.Z(2),
            ],
            6,
        ),
        _single_qubit_qiskit_circuit(),
    ),
    (
        g.Circuit(
            [
                bg.CNOT(0, 1),
            ],
            4,
        ),
        _two_qubit_qiskit_circuit(),
    ),
    (
        g.Circuit(
            [
                bg.RX(np.pi)(1),
            ],
            4,
        ),
        _parametric_qiskit_circuit(np.pi),
    ),
    (
        g.Circuit(
            [bg.SWAP.controlled(1)(2, 0, 3)],
            5,
        ),
        _qiskit_circuit_with_controlled_gate(),
    ),
    (
        g.Circuit(
            [bg.Y.controlled(2)(4, 5, 2)],
            6,
        ),
        _qiskit_circuit_with_multicontrolled_gate(),
    ),
]


EQUIVALENT_PARAMETRIZED_CIRCUITS = [
    (
        g.Circuit(
            [
                bg.RX(SYMPY_THETA)(1),
            ],
            4,
        ),
        _parametric_qiskit_circuit(QISKIT_THETA),
    ),
]


def _draw_qiskit_circuit(circuit):
    return qiskit.visualization.circuit_drawer(circuit, output="text")


class TestQiskitCircuitConversion:
    @pytest.mark.parametrize("zquantum_circuit, qiskit_circuit", EQUIVALENT_CIRCUITS)
    def test_converting_circuit_gives_equivalent_circuit(
        self, zquantum_circuit, qiskit_circuit
    ):
        converted = convert_to_qiskit(zquantum_circuit)
        assert converted == qiskit_circuit, (
            f"Converted circuit:\n{_draw_qiskit_circuit(converted)}\n isn't equal "
            f"to\n{_draw_qiskit_circuit(qiskit_circuit)}"
        )

    @pytest.mark.parametrize(
        "zquantum_circuit, qiskit_circuit", EQUIVALENT_PARAMETRIZED_CIRCUITS
    )
    def test_converting_parametrized_circuit_(self, zquantum_circuit, qiskit_circuit):
        # NOTE: parametrized circuit conversion is unsupported broken because qiskit
        # requires using singleton parameter objects
        with pytest.raises(NotImplementedError):
            convert_to_qiskit(zquantum_circuit)
