import numpy as np
import pytest
from zquantum.core.circuits import CNOT, RY, U3, Circuit, GateOperation, X, Y, Z
from zquantum.core.decompositions import U3GateToRotation, decompose_zquantum_circuit

DECOMPOSITION_RULES = [U3GateToRotation()]
U3_GATE_PARAMETERS = [
    (0, 0, 0),
    (0, np.pi, 0),
    (np.pi, 0, 0),
    (0, 0, np.pi),
    (np.pi / 2, np.pi / 2, np.pi / 2),
    (0.1 * np.pi, 0.5 * np.pi, 0.3 * np.pi),
    (4.1 * np.pi / 2, 2.5 * np.pi, 3 * np.pi),
]
U3_GATES = [U3(theta, phi, lambda_) for theta, phi, lambda_ in U3_GATE_PARAMETERS]
CU3_GATES = [
    U3(theta, phi, lambda_).controlled(1) for theta, phi, lambda_ in U3_GATE_PARAMETERS
]


def _is_scaled_identity(matrix: np.ndarray, rtol=1e-5, atol=1e-8) -> bool:
    """Returns whether 2 unitaries are equivalent up to a global phase.

    Usuage:
    _is_scaled_identity(unitary_1 @ np.linalg.inv(unitary_2))
    """
    assert matrix.shape == (
        matrix.shape[0],
        matrix.shape[0],
    ), "This test is meaningful only for square matrices"

    target_matrix = np.diag(matrix).mean() * np.eye(
        matrix.shape[0], dtype=np.complex128
    )
    return np.allclose(matrix, target_matrix, rtol, atol)


class TestDecompositionOfU3Gates:
    @pytest.mark.parametrize(
        "gate_to_be_decomposed, target_qubits",
        [
            *[(gate, qubits) for gate in U3_GATES for qubits in [(0,), (2,)]],
            *[(gate, qubits) for gate in CU3_GATES for qubits in [(0, 1), (3, 11)]],
        ],
    )
    def _test_gives_the_same_unitary_as_original_gate(
        self, gate_to_be_decomposed, target_qubits
    ):
        circuit = Circuit([gate_to_be_decomposed(*target_qubits)])
        decomposed_circuit = decompose_zquantum_circuit(circuit, DECOMPOSITION_RULES)

        assert _is_scaled_identity(
            circuit.to_unitary() @ np.linalg.inv(decomposed_circuit.to_unitary()),
            atol=0.2,
        )
        # TODO: 0.2 tolerance is too big?

    @pytest.mark.parametrize(
        "operations",
        [[RY(np.pi / 2)(0)], [X(3), Y(1), Z(0)], [CNOT(3, 11)]],
    )
    def test_leaves_gates_not_matching_predicate_unaffected(self, operations):
        circuit = Circuit(operations)
        decomposed_circuit = decompose_zquantum_circuit(circuit, DECOMPOSITION_RULES)

        assert circuit.operations == decomposed_circuit.operations

    @pytest.mark.parametrize("target_qubits", [(0,), (2,)])
    @pytest.mark.parametrize("gate_to_be_decomposed", U3_GATES)
    def test_U3_decomposition_comprises_only_rotations(
        self, gate_to_be_decomposed, target_qubits
    ):
        circuit = Circuit([gate_to_be_decomposed(*target_qubits)])
        decomposed_circuit = decompose_zquantum_circuit(circuit, DECOMPOSITION_RULES)

        assert all(
            isinstance(op, GateOperation) and op.gate.name in ("RZ", "RY")
            for op in decomposed_circuit.operations
        )

    @pytest.mark.parametrize("target_qubits", [(0, 1), (3, 11)])
    @pytest.mark.parametrize("gate_to_be_decomposed", CU3_GATES)
    def test_CU3_decomposition_comprises_only_controlled_rotations(
        self, gate_to_be_decomposed, target_qubits
    ):
        circuit = Circuit([gate_to_be_decomposed(*target_qubits)])
        decomposed_circuit = decompose_zquantum_circuit(circuit, DECOMPOSITION_RULES)

        assert all(
            isinstance(op, GateOperation)
            and op.gate.name == "Control"
            and op.gate.wrapped_gate.name in ("RZ", "RY")
            for op in decomposed_circuit.operations
        )
