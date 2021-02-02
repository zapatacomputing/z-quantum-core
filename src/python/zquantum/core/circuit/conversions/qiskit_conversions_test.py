import pytest
import qiskit
import numpy as np
from zquantum.core.circuit import X, Y, Z, I, T, H, Gate, Circuit
from .qiskit_conversions import convert_to_qiskit, convert_from_qiskit


EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES = [
    (X, qiskit.extensions.XGate),
    (Y, qiskit.extensions.YGate),
    (Z, qiskit.extensions.ZGate),
    (H, qiskit.extensions.HGate),
    (I, qiskit.extensions.IGate),
    (T, qiskit.extensions.TGate),
]


def qiskit_qubit(index: int) -> qiskit.circuit.Qubit:
    return qiskit.circuit.Qubit(qiskit.circuit.QuantumRegister(index + 1, "q"), index)


TEST_CASES_WITHOUT_SYMBOLIC_PARAMS = [
    *[
        (orquestra_gate(qubit), (qiskit_gate(), [qiskit_qubit(qubit)], []))
        for orquestra_gate, qiskit_gate in EQUIVALENT_NONPARAMETRIC_SINGLE_QUBIT_GATES
        for qubit in [0, 1, 4, 10]
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
        np.testing.assert_allclose(
            np.array(orquestra_gate.matrix).astype(np.complex128),
            qiskit_operation[0].to_matrix()
        )
