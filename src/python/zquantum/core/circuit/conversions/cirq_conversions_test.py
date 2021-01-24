import cirq
import numpy as np
import pytest
import sympy

from .cirq_conversions import convert_from_cirq, convert_to_cirq, parse_gate_name_from_cirq_gate
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

@pytest.mark.parametrize(
    "cirq_gate, expected_gate_name",
    [
        (cirq.X, "X"),
        (cirq.Y, "Y"),
        (cirq.Z, "Z"),
        (cirq.H, "H"),
        (cirq.I, "I"),
        (cirq.T, "T"),
        (cirq.rx(0.5), "Rx"),
        (cirq.rx(sympy.Symbol("theta")), "Rx"),
        (cirq.ry(np.pi), "Ry"),
        (cirq.ry(sympy.Symbol("gamma")), "Ry"),
        (cirq.rz(-np.pi / 2), "Rz"),
        (cirq.rz(sympy.Symbol("gamma")), "Rz")
    ]
)
def test_parsing_gate_name_from_cirq_gate_gives_correct_string(cirq_gate, expected_gate_name):
    assert parse_gate_name_from_cirq_gate(cirq_gate) == expected_gate_name

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


@pytest.mark.parametrize("qubit_index", [0, 4, 10, 11])
@pytest.mark.parametrize("angle", [np.pi, np.pi / 2, 0.4])
@pytest.mark.parametrize(
    "orquestra_gate_cls, cirq_func",
    [
        (RX, cirq.rx),
        (RY, cirq.ry),
        (RZ, cirq.rz),
    ],
)
class TestSingleQubitRotationGatesConversion:
    def test_conversion_from_orquestra_to_cirq_gives_correct_gate(
        self, qubit_index, angle, orquestra_gate_cls, cirq_func
    ):
        assert cirq_func(angle)(cirq.LineQubit(qubit_index)) == convert_to_cirq(
            orquestra_gate_cls(qubit_index, angle)
        )

    def test_conversion_from_cirq_to_orquestra_gives_correct_gate(
        self, qubit_index, angle, orquestra_gate_cls, cirq_func
    ):
        assert orquestra_gate_cls(qubit_index, angle) == convert_from_cirq(
            cirq_func(angle)(cirq.LineQubit(qubit_index))
        )
