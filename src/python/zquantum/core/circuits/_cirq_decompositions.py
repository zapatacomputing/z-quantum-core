from typing import Iterable, cast

import cirq

from ._decomposition import DecompositionRule, decompose_operations


def _is_cirq_rotation(gate: cirq.EigenGate):
    return gate.global_shift == -0.5


def _is_pauli(gate: cirq.Gate):
    return (
        # covers _PauliX etc.
        isinstance(gate, (type(cirq.X), type(cirq.Y), type(cirq.Z)))
        or
        # covers X ** 1 etc.
        (
            isinstance(gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate))
            and gate.exponent == 1
            and gate.global_shift == 0
        )
    )


class PowerGateToPhaseAndRotation(DecompositionRule[cirq.Operation]):
    """Decomposition of cirq's XPowGate and YPowGate."""

    def __init__(self, *gate_types: type):
        if not gate_types:
            raise ValueError(
                "No power gate type provided. You need to provide at least "
                "one of: cirq.XPowGate, cirq.YPowGate."
            )
        if any(
            gate_type not in (cirq.XPowGate, cirq.YPowGate) for gate_type in gate_types
        ):
            raise ValueError(
                "This decomposition rule supports only cirq.XPowGate and "
                f"cirq.YPowGate but {gate_types} were provided."
            )
        self.gate_types = tuple(gate_types)

    def predicate(self, operation: cirq.Operation) -> bool:
        return (
            isinstance(operation, cirq.GateOperation)
            and isinstance(operation.gate, self.gate_types)
            and not _is_cirq_rotation(cast(cirq.EigenGate, operation.gate))
            and not _is_pauli(operation.gate)
        )

    def production(self, operation: cirq.Operation) -> Iterable[cirq.Operation]:
        target_qubit = operation.qubits[0]
        original_gate = cast(cirq.EigenGate, operation.gate)
        rotation_type = type(operation.gate)
        phase_exponent = (original_gate.global_shift + 0.5) * original_gate.exponent
        return [
            # Global phase, equivalent to Identity * exp((global_shift + 0.5) * exponent)
            cirq.ZPowGate(exponent=phase_exponent).on(target_qubit),
            cirq.X.on(target_qubit),
            cirq.ZPowGate(exponent=phase_exponent).on(target_qubit),
            cirq.X.on(target_qubit),
            # Actual rotation (global_shift = -0.5) along the same axis
            rotation_type(exponent=original_gate.exponent, global_shift=-0.5).on(
                target_qubit
            ),
        ]


def decompose_cirq_circuit(
    circuit: cirq.Circuit,
    decomposition_rules: Iterable[DecompositionRule[cirq.Operation]],
):
    return cirq.Circuit(
        decompose_operations(circuit.all_operations(), decomposition_rules)
    )
