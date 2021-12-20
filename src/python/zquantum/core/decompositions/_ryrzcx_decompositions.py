from typing import Iterable

from zquantum.core.circuits._builtin_gates import RY, RZ
from zquantum.core.circuits._circuit import Circuit
from zquantum.core.circuits._gates import GateOperation
from zquantum.core.decompositions._decomposition import (
    DecompositionRule,
    decompose_operations,
)


class RXtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's RX gate in the RZRYCX gateset.

    Note that this gets rid of global phase.
    """

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose RX to RY and RZ
        return (
            operation.gate.name == "RX"
        )

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        theta, phi, lambda_ = operation.params

        decomposition = [RZ(phi), RY(theta), RZ(lambda_)]

        def preprocess_gate(gate):
            return (
                gate.controlled(operation.gate.num_control_qubits)
                if operation.gate.name == "Control"
                else gate
            )

        decomposition = [
            preprocess_gate(gate)(*operation.qubit_indices) for gate in decomposition
        ]

        return reversed(decomposition)
