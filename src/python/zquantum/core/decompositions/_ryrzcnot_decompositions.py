from typing import Iterable

import numpy as np
from zquantum.core.circuits._builtin_gates import RY, RZ
from zquantum.core.circuits._circuit import Circuit
from zquantum.core.circuits._gates import GateOperation
from zquantum.core.decompositions._decomposition import (
    DecompositionRule,
    decompose_operations,
)


class RXtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's RX gate in the RZRYCNOT gateset.

    Note that this gets rid of global phase.
    """

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose RX to RY and RZ
        return operation.gate.name == "RX"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        lambda_ = operation.params
        indices = operation.qubit_indices[0]

        return [RZ(np.pi / 2)(indices), RY(lambda_)(indices), RZ(-np.pi / 2)(indices)]
