import cirq
import numpy as np
import pytest
from zquantum.core.circuits import GateOperation, import_from_cirq
from zquantum.core.decompositions import (
    PowerGateToPhaseAndRotation,
    decompose_cirq_circuit,
)


class TestDecompositionOfPowerGates:
    @pytest.mark.parametrize("target_qubit", [cirq.LineQubit(0), cirq.LineQubit(2)])
    @pytest.mark.parametrize(
        "gate_to_be_decomposed, decomposition_rules",
        [
            (
                cirq.XPowGate(exponent=0.1),
                [PowerGateToPhaseAndRotation(cirq.XPowGate)],
            ),
            (
                cirq.XPowGate(exponent=0.1, global_shift=0.2),
                [PowerGateToPhaseAndRotation(cirq.XPowGate)],
            ),
            (
                cirq.XPowGate(exponent=0.1),
                [PowerGateToPhaseAndRotation(cirq.XPowGate, cirq.YPowGate)],
            ),
            (
                cirq.YPowGate(exponent=0.1),
                [PowerGateToPhaseAndRotation(cirq.YPowGate)],
            ),
            (
                cirq.YPowGate(exponent=0.1, global_shift=0.2),
                [PowerGateToPhaseAndRotation(cirq.YPowGate)],
            ),
        ],
    )
    def test_gives_the_same_unitary_as_original_gate(
        self, gate_to_be_decomposed, decomposition_rules, target_qubit
    ):
        circuit = cirq.Circuit([gate_to_be_decomposed.on(target_qubit)])
        decomposed_circuit = decompose_cirq_circuit(circuit, decomposition_rules)

        np.testing.assert_almost_equal(
            cirq.unitary(circuit), cirq.unitary(decomposed_circuit)
        )

    @pytest.mark.parametrize(
        "decomposition_rule, operation",
        [
            (
                PowerGateToPhaseAndRotation(cirq.XPowGate),
                cirq.rx(0.5).on(cirq.LineQubit(0)),
            ),
            (
                PowerGateToPhaseAndRotation(cirq.YPowGate),
                cirq.ry(0.5).on(cirq.LineQubit(4)),
            ),
            (
                PowerGateToPhaseAndRotation(cirq.XPowGate, cirq.YPowGate),
                cirq.rx(0.5).on(cirq.LineQubit(0)),
            ),
            (
                PowerGateToPhaseAndRotation(cirq.XPowGate, cirq.YPowGate),
                cirq.ry(0.5).on(cirq.LineQubit(4)),
            ),
        ],
    )
    def test_does_not_decompose_usual_rotation_gates(
        self, decomposition_rule, operation
    ):
        circuit = cirq.Circuit([operation])
        decomposed_circuit = decompose_cirq_circuit(circuit, [decomposition_rule])

        assert list(circuit.all_operations()) == list(
            decomposed_circuit.all_operations()
        )

    @pytest.mark.parametrize(
        "decomposition_rule, operations",
        [
            (
                PowerGateToPhaseAndRotation(cirq.XPowGate),
                [cirq.YPowGate(exponent=0.1).on(cirq.LineQubit(1))],
            ),
            (
                PowerGateToPhaseAndRotation(cirq.XPowGate),
                [
                    cirq.X.on(cirq.LineQubit(3)),
                    cirq.Y.on(cirq.LineQubit(1)),
                    cirq.Z.on(cirq.LineQubit(0)),
                ],
            ),
            (
                PowerGateToPhaseAndRotation(cirq.XPowGate),
                [
                    (cirq.X ** 1).on(cirq.LineQubit(3)),
                    (cirq.Y ** 2).on(cirq.LineQubit(1)),
                    (cirq.Z ** 1).on(cirq.LineQubit(0)),
                ],
            ),
            (
                PowerGateToPhaseAndRotation(cirq.XPowGate, cirq.YPowGate),
                [cirq.CNOT.on(cirq.LineQubit(3), cirq.LineQubit(11))],
            ),
        ],
    )
    def test_leaves_gates_not_matching_predicate_unaffected(
        self, decomposition_rule, operations
    ):
        circuit = cirq.Circuit(operations)
        decomposed_circuit = decompose_cirq_circuit(circuit, [decomposition_rule])

        assert list(circuit.all_operations()) == list(
            decomposed_circuit.all_operations()
        )

    @pytest.mark.parametrize(
        "gate_to_be_decomposed",
        [
            cirq.XPowGate(exponent=0.1),
            cirq.XPowGate(exponent=0.1, global_shift=0.2),
            cirq.XPowGate(exponent=0.1),
            cirq.YPowGate(exponent=0.1),
            cirq.YPowGate(exponent=0.1, global_shift=0.2),
        ],
    )
    @pytest.mark.parametrize("target_qubit", [cirq.LineQubit(0), cirq.LineQubit(2)])
    def test_comprises_only_phase_pauli_and_rotations(
        self, gate_to_be_decomposed, target_qubit
    ):
        cirq_circuit = cirq.Circuit([gate_to_be_decomposed.on(target_qubit)])
        zquantum_circuit = import_from_cirq(
            decompose_cirq_circuit(
                cirq_circuit,
                [PowerGateToPhaseAndRotation(cirq.XPowGate, cirq.YPowGate)],
            )
        )

        assert all(
            isinstance(op, GateOperation) and op.gate.name in ("X", "PHASE", "RX", "RY")
            for op in zquantum_circuit.operations
        )

    def test_accepts_only_xpowgate_or_ypowgate_in_initializer_argument(self):
        with pytest.raises(ValueError):
            PowerGateToPhaseAndRotation(cirq.ZPowGate, cirq.XPowGate)

    def test_requires_at_least_one_gate_class_in_initializer_argument(self):
        with pytest.raises(ValueError):
            PowerGateToPhaseAndRotation()
