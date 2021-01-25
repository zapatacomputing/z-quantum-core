import cirq
import numpy as np
import pytest
import sympy

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


EXAMPLE_SYMBOLIC_ANGLES = [
    sympy.Symbol("theta"),
    sympy.Symbol("x") + sympy.Symbol("y"),
    sympy.cos(sympy.Symbol("phi") / 2)
]


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


@pytest.mark.parametrize("qubit_index", [0, 4, 10, 11])
@pytest.mark.parametrize("angle", EXAMPLE_SYMBOLIC_ANGLES)
@pytest.mark.parametrize(
    "orquestra_gate_cls, cirq_gate_func",
    [
        (RX, cirq.rx),
        (RY, cirq.ry),
        (RZ, cirq.rz),
    ],
)
class TestSingleQubitRotationGatesWithSymbolicParamsConversion:

    def test_conversion_from_orquestra_to_pyquil_gives_correct_gate(
        self,
        qubit_index,
        angle,
        orquestra_gate_cls,
        cirq_gate_func,
    ):
        assert (
            cirq_gate_func(angle)(cirq.LineQubit(qubit_index)) ==
            convert_to_cirq(orquestra_gate_cls(qubit_index, angle))
        )

    def test_conversion_from_cirq_to_orquestra_gives_correct_gate(
        self,
        qubit_index,
        angle,
        orquestra_gate_cls,
        cirq_gate_func,
    ):
        assert orquestra_gate_cls(qubit_index, angle) == convert_from_cirq(
            cirq_gate_func(angle)(cirq.LineQubit(qubit_index))
        )
