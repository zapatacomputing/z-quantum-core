################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Iterable, Sequence

from zquantum.core.circuits._builtin_gates import RY, RZ
from zquantum.core.circuits._circuit import Circuit
from zquantum.core.circuits._gates import ControlledGate, GateOperation
from zquantum.core.decompositions._decomposition import (
    DecompositionRule,
    decompose_operations,
)


class U3GateToRotation(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's U3 gate.

    Note that this gets rid of global phase.
    """

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose U3 and its controlled version
        return (
            operation.gate.name == "U3"
            or isinstance(operation.gate, ControlledGate)
            and operation.gate.wrapped_gate.name == "U3"
        )

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        theta, phi, lambda_ = operation.params

        gate_decomposition = [RZ(phi), RY(theta), RZ(lambda_)]

        def preprocess_gate(gate):
            return (
                gate.controlled(operation.gate.num_control_qubits)
                if operation.gate.name == "Control"
                else gate
            )

        gate_operation_decomposition = [
            preprocess_gate(gate)(*operation.qubit_indices)
            for gate in gate_decomposition
        ]

        return reversed(gate_operation_decomposition)


def decompose_zquantum_circuit(
    circuit: Circuit, decomposition_rules: Sequence[DecompositionRule[GateOperation]]
):
    return Circuit(decompose_operations(circuit.operations, decomposition_rules))
