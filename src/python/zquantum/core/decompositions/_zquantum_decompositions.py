from typing import Iterable

from zquantum.core.circuits._builtin_gates import RY, RZ
from zquantum.core.circuits._circuit import Circuit
from zquantum.core.circuits._gates import GateOperation
from zquantum.core.decompositions._decomposition import (
    DecompositionRule,
    decompose_operations,
)


class U3GateToRotation(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's U3 gate."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose U3 and its controlled version
        return operation.gate.name == "U3" or operation.gate.wrapped_gate.name == "U3"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:

        # How to determine if controlled?
        # Two ways: check for presence of "wrapped gate" attribute
        # using the length of qubit_indices (would be 2 for control)

        # Problem: no efficient way for unpacking the tuple (?)

        if hasattr(operation.gate, "wrapped_gate"):
            control, target = operation.qubit_indices
        else:
            (target,) = operation.qubit_indices

        theta, phi, lambda_ = operation.params

        decomposition = [RZ(phi)(target), RY(theta)(target), RZ(lambda_)(target)]

        if hasattr(operation.gate, "wrapped_gate"):
            decomposition = [
                gate.controlled(1)(control, target) for gate in decomposition
            ]

        return decomposition


def decompose_zquantum_circuit(
    circuit: Circuit, decomposition_rules: Iterable[DecompositionRule[GateOperation]]
):
    return Circuit(decompose_operations(circuit.operations, decomposition_rules))
