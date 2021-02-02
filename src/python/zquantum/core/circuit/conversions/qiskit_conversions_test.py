import pytest
import qiskit
import numpy as np
from zquantum.core.circuit import X, Y, Z, I, T, H, Gate, Circuit, CNOT, CZ, SWAP, ISWAP
from .qiskit_conversions import convert_to_qiskit, convert_from_qiskit


EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES = [
    (X, qiskit.extensions.XGate),
    (Y, qiskit.extensions.YGate),
    (Z, qiskit.extensions.ZGate),
    (H, qiskit.extensions.HGate),
    (I, qiskit.extensions.IGate),
    (T, qiskit.extensions.TGate),
]


EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES = [
    (CNOT, qiskit.extensions.CnotGate),
    (CZ, qiskit.extensions.CZGate),
    (SWAP, qiskit.extensions.SwapGate),
    (ISWAP, qiskit.extensions.iSwapGate),
]


TWO_QUBIT_SWAP_MATRIX = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)


def qiskit_qubit(index: int) -> qiskit.circuit.Qubit:
    return qiskit.circuit.Qubit(qiskit.circuit.QuantumRegister(index + 1, "q"), index)


TEST_CASES_WITHOUT_SYMBOLIC_PARAMS = [
    *[
        (orquestra_gate(qubit), (qiskit_gate(), [qiskit_qubit(qubit)], []))
        for orquestra_gate, qiskit_gate in EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES
        for qubit in [0, 1, 4, 10]
    ],
    *[
        (
            orquestra_gate(*qubits),
            (qiskit_gate(), [qiskit_qubit(qubit) for qubit in reversed(qubits)], []),
        )
        for orquestra_gate, qiskit_gate in EQUIVALENT_NONPARAMETRIC_TWO_QUBIT_GATES
        for qubits in [(0, 1), (3, 4), (10, 1)]
    ],
]


@pytest.mark.parametrize(
    "orquestra_gate, qiskit_operation", TEST_CASES_WITHOUT_SYMBOLIC_PARAMS
)
class TestGateConversionWithoutSymbolicParameters:
    def test_converting_orquestra_gate_to_qiskit_gives_expected_operation(
        self, orquestra_gate, qiskit_operation
    ):
        assert (
            convert_to_qiskit(orquestra_gate, orquestra_gate.qubits[0] + 1)
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
