from typing import Iterable

from zquantum.core.circuits._builtin_gates import RY, RZ
from zquantum.core.circuits._circuit import Circuit
from zquantum.core.circuits._gates import GateOperation
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
            or hasattr(operation.gate, "wrapped_gate")
            and operation.gate.wrapped_gate.name == "U3"
        )

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:

        # How to determine if controlled?
        # Two ways: check for presence of "wrapped gate" attribute
        # using the length of qubit_indices (would be 2 for control)

        # Problem: no efficient way for unpacking the tuple (?)

        theta, phi, lambda_ = operation.params

        decomposition = [RZ(phi), RY(theta), RZ(lambda_)]

        if hasattr(operation.gate, "wrapped_gate"):
            # operation.gate is controlled U3
            num_controlled_qubits = operation.gate.num_control_qubits
            decomposition = [
                gate.controlled(num_controlled_qubits)(*operation.qubit_indices)
                for gate in decomposition
            ]
        else:
            decomposition = [gate(*operation.qubit_indices) for gate in decomposition]

        return decomposition


def decompose_zquantum_circuit(
    circuit: Circuit, decomposition_rules: Iterable[DecompositionRule[GateOperation]]
):
    return Circuit(decompose_operations(circuit.operations, decomposition_rules))
