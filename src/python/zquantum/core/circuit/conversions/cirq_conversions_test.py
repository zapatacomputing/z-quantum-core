import cirq
import numpy as np
import pytest
import sympy

from .cirq_conversions import convert_from_cirq, convert_to_cirq, make_rotation_factory
from .. import XY
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
    XX,
    YY,
    ZZ,
)


EXAMPLE_SYMBOLIC_ANGLES = [
    sympy.Symbol("theta"),
    sympy.Symbol("x") + sympy.Symbol("y"),
    sympy.cos(sympy.Symbol("phi") / 2),
]


EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES = [
    (X, cirq.X),
    (Y, cirq.Y),
    (Z, cirq.Z),
    (T, cirq.T),
    (I, cirq.I),
    (H, cirq.H),
]


EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES = [
    (RX, cirq.rx),
    (RY, cirq.ry),
    (RZ, cirq.rz),
    # There is no PHASE gate in cirq, so the pair below is a bit of cheating
    # so we can fit into tests that follow.
    (PHASE, make_rotation_factory(cirq.ZPowGate)),
]


EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES = [
    (CZ, cirq.CZ),
    (CNOT, cirq.CNOT),
    (SWAP, cirq.SWAP),
]


TWO_QUBIT_ROTATION_GATE_FACTORIES = [
    (CPHASE, make_rotation_factory(cirq.CZPowGate)),
    (XX, make_rotation_factory(cirq.XXPowGate, global_shift=-0.5)),
    (YY, make_rotation_factory(cirq.YYPowGate, global_shift=-0.5)),
    (ZZ, make_rotation_factory(cirq.ZZPowGate, global_shift=-0.5)),
    (XY, make_rotation_factory(cirq.ISwapPowGate, 0.0))
]


# Here we combine multiple testcases of the form
# (Orquestra gate, Cirq operation)
# We do this for easier parametrization in tests that follow.
TEST_CASES_WITHOUT_SYMBOLIC_PARAMS = (
    [
        (orq_gate_cls(q), cirq_gate.on(cirq.LineQubit(q)))
        for orq_gate_cls, cirq_gate in EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES
        for q in [0, 1, 5, 13]
    ]
    + [
        (orq_gate_cls(q0, q1), cirq_gate.on(cirq.LineQubit(q0), cirq.LineQubit(q1)))
        for orq_gate_cls, cirq_gate in EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES
        for q0, q1 in [(0, 1), (2, 3), (0, 10)]
    ]
    + [
        (orq_gate_cls(q, angle), cirq_gate_func(angle).on(cirq.LineQubit(q)))
        for orq_gate_cls, cirq_gate_func in EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES
        for q in [0, 4, 10, 11]
        for angle in [np.pi, np.pi / 2, 0.4]
    ]
    + [
        (
            orq_gate_cls(q0, q1, angle),
            cirq_gate_func(angle).on(cirq.LineQubit(q0), cirq.LineQubit(q1)),
        )
        for orq_gate_cls, cirq_gate_func in TWO_QUBIT_ROTATION_GATE_FACTORIES
        for q0, q1 in [(0, 1), (2, 3), (0, 10)]
        for angle in [np.pi, np.pi / 2, np.pi / 5, 0.4, 0.1, 0.05, 2.5]
    ]
)


TEST_CASES_WITH_SYMBOLIC_PARAMS = [
    (orq_gate_cls(q, angle), cirq_gate_func(angle).on(cirq.LineQubit(q)))
    for orq_gate_cls, cirq_gate_func in EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES
    for q in [0, 4, 10, 11]
    for angle in EXAMPLE_SYMBOLIC_ANGLES
] + [
    (
        orq_gate_cls(q0, q1, angle),
        cirq_gate_func(angle).on(cirq.LineQubit(q0), cirq.LineQubit(q1)),
    )
    for orq_gate_cls, cirq_gate_func in TWO_QUBIT_ROTATION_GATE_FACTORIES
    for q0, q1 in [(0, 1), (2, 3), (0, 10)]
    for angle in EXAMPLE_SYMBOLIC_ANGLES
]


@pytest.mark.parametrize(
    "orquestra_gate, cirq_operation", TEST_CASES_WITHOUT_SYMBOLIC_PARAMS
)
class TestGateConversionWithoutSymbolicParameters:
    def test_converting_orquestra_gate_to_cirq_gives_expected_operation(
        self, orquestra_gate, cirq_operation
    ):
        assert convert_to_cirq(orquestra_gate) == cirq_operation

    def test_converting_cirq_operation_to_orquestra_gives_expected_gate(
        self, orquestra_gate, cirq_operation
    ):
        assert convert_from_cirq(cirq_operation) == orquestra_gate

    def test_orquestra_gate_and_cirq_gate_have_the_same_matrix(
        self, orquestra_gate, cirq_operation
    ):
        # This is to ensure that we are indeed converting the same gate.
        assert np.allclose(
            np.array(orquestra_gate.matrix).astype(np.complex128),
            cirq.unitary(cirq_operation.gate),
        )


@pytest.mark.parametrize(
    "orquestra_gate, cirq_operation", TEST_CASES_WITH_SYMBOLIC_PARAMS
)
class TestGateConversionWithSymbolicParameters:
    def test_converting_orquestra_gate_to_cirq_gives_expected_operation(
        self, orquestra_gate, cirq_operation
    ):
        assert convert_to_cirq(orquestra_gate) == cirq_operation

    def test_converting_cirq_operation_to_orquestra_gives_expected_gate(
        self, orquestra_gate, cirq_operation
    ):
        assert convert_from_cirq(cirq_operation) == orquestra_gate
