"""Test cases from Orquestra <-> cirq conversions."""
import cirq
import numpy as np
import pytest
import sympy

from .cirq_conversions import convert_from_cirq, convert_to_cirq, make_rotation_factory
from .. import _builtin_gates as bg


THETA = sympy.Symbol("theta")

EXAMPLE_SYMBOLIC_ANGLES = [
    sympy.Symbol("theta"),
    sympy.Symbol("x") + sympy.Symbol("y"),
    sympy.cos(sympy.Symbol("phi") / 2),
]


EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES = [
    (bg.X, cirq.X),
    (bg.Y, cirq.Y),
    (bg.Z, cirq.Z),
    (bg.T, cirq.T),
    (bg.I, cirq.I),
    (bg.H, cirq.H),
]


EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES = [
    (bg.RX, cirq.rx),
    (bg.RY, cirq.ry),
    (bg.RZ, cirq.rz),
    (bg.PHASE, make_rotation_factory(cirq.ZPowGate)),
]


EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES = [
    (bg.CZ, cirq.CZ),
    (bg.CNOT, cirq.CNOT),
    (bg.SWAP, cirq.SWAP),
]


TWO_QUBIT_ROTATION_GATE_FACTORIES = [
    (bg.CPHASE, make_rotation_factory(cirq.CZPowGate)),
    (bg.XX, make_rotation_factory(cirq.XXPowGate, global_shift=-0.5)),
    (bg.YY, make_rotation_factory(cirq.YYPowGate, global_shift=-0.5)),
    (bg.ZZ, make_rotation_factory(cirq.ZZPowGate, global_shift=-0.5)),
    (bg.XY, make_rotation_factory(cirq.ISwapPowGate, 0.0))
]


# Here we combine multiple testcases of the form
# (ZQuantum gate, Cirq operation)
# We do this for easier parametrization in tests that follow.
TEST_CASES_WITHOUT_SYMBOLIC_PARAMS = (
    [
        (orq_gate(q), cirq_gate.on(cirq.LineQubit(q)))
        for orq_gate, cirq_gate in EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES
        for q in [0, 1, 5, 13]
    ]
    + [
        (orq_gate(q0, q1), cirq_gate.on(cirq.LineQubit(q0), cirq.LineQubit(q1)))
        for orq_gate, cirq_gate in EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES
        for q0, q1 in [(0, 1), (2, 3), (0, 10)]
    ]
    + [
        (orq_gate_factory(angle)(q), cirq_gate_func(angle).on(cirq.LineQubit(q)))
        for orq_gate_factory, cirq_gate_func in EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES
        for q in [0, 4, 10, 11]
        for angle in [0, np.pi, np.pi / 2, 0.4]
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
    (orq_gate_factory(angle)(q), cirq_gate_func(angle).on(cirq.LineQubit(q)))
    for orq_gate_factory, cirq_gate_func in EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES
    for q in [0, 4, 10, 11]
    for angle in EXAMPLE_SYMBOLIC_ANGLES
] + [
    (
        orq_gate_factory(angle)(q0, q1),
        cirq_gate_func(angle).on(cirq.LineQubit(q0), cirq.LineQubit(q1)),
    )
    for orq_gate_factory, cirq_gate_func in TWO_QUBIT_ROTATION_GATE_FACTORIES
    for q0, q1 in [(0, 1), (2, 3), (0, 10)]
    for angle in EXAMPLE_SYMBOLIC_ANGLES
]


@pytest.mark.parametrize(
    "orquestra_operation, cirq_operation", TEST_CASES_WITHOUT_SYMBOLIC_PARAMS
)
class TestGateConversionWithoutSymbolicParameters:
    def test_converting_orquestra_gate_operation_to_cirq_gives_expected_operation(
        self, orquestra_operation, cirq_operation
    ):
        assert convert_to_cirq(orquestra_operation) == cirq_operation

    def test_converting_cirq_operation_to_orquestra_gives_expected_gate_operation(
        self, orquestra_operation, cirq_operation
    ):
        assert convert_from_cirq(cirq_operation) == orquestra_operation

    def test_orquestra_gate_and_cirq_gate_have_the_same_matrix(
        self, orquestra_operation, cirq_operation
    ):
        # This is to ensure that we are indeed converting the same gate.
        assert np.allclose(
            np.array(orquestra_operation.gate.matrix).astype(np.complex128),
            cirq.unitary(cirq_operation.gate),
        )


@pytest.mark.parametrize(
    "orquestra_operation, cirq_operation", TEST_CASES_WITH_SYMBOLIC_PARAMS
)
class TestGateConversionWithSymbolicParameters:
    def test_converting_orquestra_gate_operation_to_cirq_gives_expected_operation(
        self, orquestra_operation, cirq_operation
    ):
        assert convert_to_cirq(orquestra_operation) == cirq_operation

    def test_converting_cirq_operation_to_orquestra_gives_expected_gate_operation(
        self, orquestra_operation, cirq_operation
    ):
        assert convert_from_cirq(cirq_operation) == orquestra_operation
