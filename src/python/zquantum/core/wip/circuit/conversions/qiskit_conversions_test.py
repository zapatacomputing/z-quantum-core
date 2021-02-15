import qiskit
import numpy as np
import sympy
import pytest

from .. import (
    X,
    Y,
    Z,
    I,
    T,
    H,
    CNOT,
    CZ,
    SWAP,
    ISWAP,
    RX,
    RY,
    RZ,
    PHASE,
    CPHASE,
    XX,
    YY,
    ZZ,
    Circuit,
)
from .qiskit_conversions import convert_to_qiskit, convert_from_qiskit, qiskit_qubit


EXAMPLE_SYMBOLIC_ANGLES = [
    (sympy.Symbol("theta"), qiskit.circuit.Parameter("theta")),
    (
        sympy.Symbol("x") + sympy.Symbol("y"),
        qiskit.circuit.Parameter("x") + qiskit.circuit.Parameter("y"),
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
    (PHASE, qiskit.extensions.PhaseGate),
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
            orquestra_gate(*qubit_pair),
            (
                qiskit_gate(),
                [qiskit_qubit(qubit, max(qubit_pair) + 1) for qubit in qubit_pair],
                [],
            ),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES
        for qubit_pair in [(0, 1), (3, 4), (10, 1)]
    ],
    *[
        (
            orquestra_gate(qubit, angle),
            (qiskit_gate(angle), [qiskit_qubit(qubit, qubit + 1)], []),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES
        for qubit in [0, 1, 4, 10]
        for angle in [0, np.pi, np.pi / 2, 0.4, np.pi / 5]
    ],
    *[
        (
            orquestra_gate(*qubit_pair, angle),
            (
                qiskit_gate(angle),
                [qiskit_qubit(qubit, max(qubit_pair) + 1) for qubit in qubit_pair],
                [],
            ),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_TWO_QUBIT_ROTATION_GATES
        for qubit_pair in [(0, 1), (3, 4), (10, 1)]
        for angle in [0, np.pi, np.pi / 2, 0.4, np.pi / 5]
    ],
]


TEST_CASES_WITH_SYMBOLIC_PARAMS = [
    *[
        (
            orquestra_gate(qubit, orquestra_angle),
            (qiskit_gate(qiskit_angle), [qiskit_qubit(qubit, qubit + 1)], []),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_SINGLE_QUBIT_ROTATION_GATES
        for qubit in [0, 1, 4, 10]
        for orquestra_angle, qiskit_angle in EXAMPLE_SYMBOLIC_ANGLES
    ],
    *[
        (
            orquestra_gate(*qubit_pair, orquestra_angle),
            (
                qiskit_gate(qiskit_angle),
                [qiskit_qubit(qubit, max(qubit_pair) + 1) for qubit in qubit_pair],
                [],
            ),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_TWO_QUBIT_ROTATION_GATES
        for qubit_pair in [(0, 1), (3, 4), (10, 1)]
        for orquestra_angle, qiskit_angle in EXAMPLE_SYMBOLIC_ANGLES
    ],
]


# NOTE: In Qiskit, 0 is the most significant qubit,
# whereas in Orquestra, 0 is the least significant qubit.
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


def _parametric_qiskit_circuit():
    qc = qiskit.QuantumCircuit(4)
    qc.rx(np.pi, 0)
    return qc


EQUIVALENT_CIRCUITS = [
    (
        Circuit(
            [
                X(0),
                Z(2),
            ],
            6,
        ),
        _single_qubit_qiskit_circuit(),
    ),
    (
        Circuit(
            [
                CNOT(0, 1),
            ],
            4,
        ),
        _two_qubit_qiskit_circuit(),
    ),
    (
        Circuit(
            [
                RX(0, np.pi),
            ],
            4,
        ),
        _parametric_qiskit_circuit(),
    ),
]


def are_qiskit_parameters_equal(param_1, param_2):
    return (
        getattr(param_1, "_symbol_expr", param_1)
        - getattr(param_2, "_symbol_expr", param_2)
        == 0
    )


def are_qiskit_gates_equal(gate_1, gate_2):
    type_1, type_2 = type(gate_1), type(gate_2)
    return (issubclass(type_1, type_2) or issubclass(type_2, type_1)) and all(
        are_qiskit_parameters_equal(param_1, param_2)
        for param_1, param_2 in zip(gate_1.params, gate_2.params)
    )


def _are_qiskit_operations_equal(operation_1, operation_2):
    return operation_1[1:] == operation_2[1:] and are_qiskit_gates_equal(
        operation_1[0], operation_2[0]
    )


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

    def test_converting_qiskit_operation_to_orquestra_gives_expected_gate(
        self, orquestra_gate, qiskit_operation
    ):
        assert convert_from_qiskit(qiskit_operation) == orquestra_gate

    def test_orquestra_gate_and_qiskit_gate_have_the_same_matrix(
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
        assert _are_qiskit_operations_equal(
            convert_to_qiskit(orquestra_gate, max(orquestra_gate.qubits) + 1),
            qiskit_operation,
        )

    def test_converting_qiskit_operation_to_orquestra_gives_expected_gate(
        self, orquestra_gate, qiskit_operation
    ):
        assert convert_from_qiskit(qiskit_operation) == orquestra_gate


def _draw_qiskit_circuit(circuit):
    return qiskit.visualization.circuit_drawer(circuit, output="text")


@pytest.mark.parametrize("orquestra_circuit, qiskit_circuit", EQUIVALENT_CIRCUITS)
class TestCircuitConversion:
    def test_converting_orquestra_circuit_to_qiskit_gives_expected_circuit(
        self, orquestra_circuit, qiskit_circuit
    ):
        converted = convert_to_qiskit(orquestra_circuit)
        assert converted == qiskit_circuit, (
            f"Converted circuit:\n{_draw_qiskit_circuit(converted)}\n isn't equal "
            f"to {_draw_qiskit_circuit(qiskit_circuit)}"
        )
