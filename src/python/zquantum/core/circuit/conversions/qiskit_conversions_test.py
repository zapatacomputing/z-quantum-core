import pytest
import qiskit
import numpy as np
import sympy
from zquantum.core.circuit import X, Y, Z, I, T, H, Gate, Circuit, CNOT, CZ, SWAP, ISWAP, RX, RY, RZ, PHASE, CPHASE, XX, \
    YY, ZZ, XY
from .qiskit_conversions import convert_to_qiskit, convert_from_qiskit, qiskit_qubit


EXAMPLE_SYMBOLIC_ANGLES = [
    (sympy.Symbol("theta"), qiskit.circuit.Parameter("theta")),
    (
        sympy.Symbol("x") + sympy.Symbol("y"),
        qiskit.circuit.Parameter("x") + qiskit.circuit.Parameter("y")
    ),
    (0.5 * sympy.Symbol("phi") + 1, 0.5 * qiskit.circuit.Parameter("phi") + 1),
]


EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES = [
    (X, qiskit.extensions.XGate),
    (Y, qiskit.extensions.YGate),
    (Z, qiskit.extensions.ZGate),
    (H, qiskit.extensions.HGate),
    (I, qiskit.extensions.IGate),
    (T, qiskit.extensions.TGate),
]


EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES = [
    # TODO: use non-deprecated CXGate
    (CNOT, qiskit.extensions.CXGate),
    (CZ, qiskit.extensions.CZGate),
    (SWAP, qiskit.extensions.SwapGate),
    (ISWAP, qiskit.extensions.iSwapGate),
]


EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES = [
    (RX, qiskit.extensions.RXGate),
    (RY, qiskit.extensions.RYGate),
    (RZ, qiskit.extensions.RZGate),
    (PHASE, qiskit.extensions.PhaseGate)
]


EQUIVALENT_TWO_QUBIT_ROTATION_GATES = [
    (CPHASE, qiskit.extensions.CPhaseGate),
    (XX, qiskit.extensions.RXXGate),
    (YY, qiskit.extensions.RYYGate),
    (ZZ, qiskit.extensions.RZZGate),
]


TWO_QUBIT_SWAP_MATRIX = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)


TEST_CASES_WITHOUT_SYMBOLIC_PARAMS = [
    *[
        (orquestra_gate(qubit), (qiskit_gate(), [qiskit_qubit(qubit, qubit + 1)], []))
        for orquestra_gate, qiskit_gate in EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES
        for qubit in [0, 1, 4, 10]
    ],
    *[
        (
            orquestra_gate(*qubits),
            (
                qiskit_gate(),
                [qiskit_qubit(qubit, max(qubits) + 1) for qubit in reversed(qubits)],
                [],
            ),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES
        for qubits in [(0, 1), (3, 4), (10, 1)]
    ],
    *[
        (orquestra_gate(qubit, angle), (qiskit_gate(angle), [qiskit_qubit(qubit, qubit + 1)], []))
        for orquestra_gate, qiskit_gate in EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES
        for qubit in [0, 1, 4, 10]
        for angle in [0, np.pi, np.pi / 2, 0.4, np.pi/5]
    ],
    *[
        (
            orquestra_gate(*qubits, angle),
            (
                qiskit_gate(angle),
                [qiskit_qubit(qubit, max(qubits) + 1) for qubit in reversed(qubits)],
                [],
            ),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_TWO_QUBIT_ROTATION_GATES
        for qubits in [(0, 1), (3, 4), (10, 1)]
        for angle in [0, np.pi, np.pi / 2, 0.4, np.pi/5]
    ]
]


TEST_CASES_WITH_SYMBOLIC_PARAMS = [
    *[
        (orquestra_gate(qubit, orquestra_angle), (qiskit_gate(qiskit_angle), [qiskit_qubit(qubit, qubit + 1)], []))
        for orquestra_gate, qiskit_gate in EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES
        for qubit in [0, 1, 4, 10]
        for orquestra_angle, qiskit_angle in EXAMPLE_SYMBOLIC_ANGLES
    ],
    *[
        (
            orquestra_gate(*qubits, orquestra_angle),
            (
                qiskit_gate(qiskit_angle),
                [qiskit_qubit(qubit, max(qubits) + 1) for qubit in reversed(qubits)],
                [],
            ),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_TWO_QUBIT_ROTATION_GATES
        for qubits in [(0, 1), (3, 4), (10, 1)]
        for orquestra_angle, qiskit_angle in EXAMPLE_SYMBOLIC_ANGLES
    ]
]


@pytest.mark.parametrize(
    "orquestra_gate, qiskit_operation", TEST_CASES_WITHOUT_SYMBOLIC_PARAMS
)
class TestGateConversionWithoutSymbolicParameters:
    def test_converting_orquestra_gate_to_qiskit_gives_expected_operation(
        self, orquestra_gate, qiskit_operation
    ):
        assert (
            convert_to_qiskit(orquestra_gate, max(orquestra_gate.qubits) + 1)
            == qiskit_operation
        )

    def test_converting_cirq_operation_to_orquestra_gives_expected_gate(
        self, orquestra_gate, qiskit_operation
    ):
        assert convert_from_qiskit(qiskit_operation) == orquestra_gate

    def test_orquestra_gate_and_cirq_gate_have_the_same_matrix(
        self, orquestra_gate, qiskit_operation
    ):
        orquestra_matrix = np.array(orquestra_gate.matrix).astype(np.complex128)
        if len(orquestra_gate.qubits) == 2:
            orquestra_matrix = (
                TWO_QUBIT_SWAP_MATRIX @ orquestra_matrix @ TWO_QUBIT_SWAP_MATRIX
            )
        np.testing.assert_allclose(orquestra_matrix, qiskit_operation[0].to_matrix())


class TestQiskitQubit:
    def test_qiskit_qubit_produces_qubit_with_specified_index(self):
        qubit = qiskit_qubit(0, 3)
        assert qubit.index == 0

    def test_qiskit_qubit_produces_qubit_with_register_having_specified_size(self):
        qubit = qiskit_qubit(1, 4)
        assert qubit.register.size == 4


@pytest.mark.parametrize(
    "orquestra_gate, qiskit_operation", TEST_CASES_WITH_SYMBOLIC_PARAMS
)
class TestGateConversionWithSymbolicParameters:

    def test_converting_orquestra_gate_to_qiskit_gives_expected_operation(
        self, orquestra_gate, qiskit_operation
    ):
        assert (
            convert_to_qiskit(orquestra_gate, max(orquestra_gate.qubits) + 1)
            == qiskit_operation
        )

    def test_converting_cirq_operation_to_orquestra_gives_expected_gate(
        self, orquestra_gate, qiskit_operation
    ):
        assert convert_from_qiskit(qiskit_operation) == orquestra_gate

    def test_orquestra_gate_and_cirq_gate_have_the_same_matrix(
        self, orquestra_gate, qiskit_operation
    ):
        orquestra_matrix = np.array(orquestra_gate.matrix).astype(np.complex128)
        if len(orquestra_gate.qubits) == 2:
            orquestra_matrix = (
                TWO_QUBIT_SWAP_MATRIX @ orquestra_matrix @ TWO_QUBIT_SWAP_MATRIX
            )
        np.testing.assert_allclose(orquestra_matrix, qiskit_operation[0].to_matrix())
