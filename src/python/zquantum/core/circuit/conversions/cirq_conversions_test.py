import cirq
import pytest

from .cirq_conversions import convert_from_cirq, convert_to_cirq
from ...circuit.gates import (
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    PHASE,
    T,
    I,
    H,
    CZ,
    CNOT,
    CPHASE,
    SWAP,
    Dagger,
)


@pytest.mark.parametrize("qubit_index", [0, 1, 5, 13])
@pytest.mark.parametrize(
    "orquestra_gate_cls, cirq_gate",
    [
        (X, cirq.X),
        (Y, cirq.Y),
        (Z, cirq.Z),
        (T, cirq.T),
        (I, cirq.I),
        (H, cirq.H),
    ],
)
class TestSingleQubitNonParametricGatesConversion:
    def test_conversion_from_orquestra_to_cirq_gives_correct_gate(
        self, qubit_index, orquestra_gate_cls, cirq_gate
    ):
        assert convert_to_cirq(orquestra_gate_cls(qubit_index)) == cirq_gate(
            cirq.LineQubit(qubit_index)
        )

    def test_conversion_from_cirq_to_orquestra_gives_correct_gate(
        self, qubit_index, orquestra_gate_cls, cirq_gate
    ):
        assert convert_from_cirq(
            cirq_gate(cirq.LineQubit(qubit_index))
        ) == orquestra_gate_cls(qubit_index)
