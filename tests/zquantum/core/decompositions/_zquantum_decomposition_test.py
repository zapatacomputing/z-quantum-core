import numpy as np
import pytest
from zquantum.core.circuits._builtin_gates import U3
from zquantum.core.circuits._circuit import Circuit
from zquantum.core.decompositions import U3GateToRotation, decompose_zquantum_circuit

DECOMPOSITION_RULES = [U3GateToRotation()]


class TestDecompositionOfPowerGates:
    @pytest.mark.parametrize("target_qubit", [0, 2])
    @pytest.mark.parametrize(
        "gate_to_be_decomposed",
        [
            U3(theta, phi, lambda_)
            for theta, phi, lambda_ in [
                (0, 0, 0),
                (0, np.pi, 0),
                # (np.pi, 0, 0),
                # (0, 0, np.pi),
                # (np.pi / 2, np.pi / 2, np.pi / 2),
                # (0.1 * np.pi, 0.5 * np.pi, 0.3 * np.pi),
                # (4.1 * np.pi / 2, 2.5 * np.pi, 3 * np.pi),
            ]
        ],
    )
    def test_gives_the_same_unitary_as_original_gate(
        self, gate_to_be_decomposed, target_qubit
    ):
        circuit = Circuit([gate_to_be_decomposed(target_qubit)])
        decomposed_circuit = decompose_zquantum_circuit(circuit, DECOMPOSITION_RULES)

        np.testing.assert_almost_equal(
            circuit.to_unitary(), decomposed_circuit.to_unitary()
        )

    # @pytest.mark.parametrize(
    #     "decomposition_rule, operations",
    #     [
    #         # (
    #         #     PowerGateToPhaseAndRotation(cirq.XPowGate),
    #         #     [cirq.YPowGate(exponent=0.1).on(cirq.LineQubit(1))],
    #         # ),
    #     ],
    # )
    # def test_leaves_gates_not_matching_predicate_unaffected(
    #     self, decomposition_rule, operations
    # ):
    #     # circuit = cirq.Circuit(operations)
    #     # decomposed_circuit = decompose_cirq_circuit(circuit, [decomposition_rule])

    #     assert list(circuit.all_operations()) == list(
    #         decomposed_circuit.all_operations()
    #     )

    # @pytest.mark.parametrize(
    #     "gate_to_be_decomposed",
    #     [

    #     ],
    # )
    # @pytest.mark.parametrize("target_qubit", [])
    # def test_comprises_only_phase_pauli_and_rotations(
    #     self, gate_to_be_decomposed, target_qubit
    # ):
    #     # cirq_circuit = cirq.Circuit([gate_to_be_decomposed.on(target_qubit)])
    #     # zquantum_circuit = import_from_cirq(
    #     #     decompose_cirq_circuit(
    #     #         cirq_circuit,
    #     #         [PowerGateToPhaseAndRotation(cirq.XPowGate, cirq.YPowGate)],
    #     #     )
    #     # )

    #     # assert all(
    #     #     isinstance(op, GateOperation)
    # and op.gate.name in ("X", "PHASE", "RX", "RY")
    #     #     for op in zquantum_circuit.operations
    #     # )
