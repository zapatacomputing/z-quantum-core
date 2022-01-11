from typing import Iterable

import numpy as np
from zquantum.core.circuits._builtin_gates import GPHASE, RY, RZ
from zquantum.core.circuits._gates import GateOperation
from zquantum.core.decompositions._decomposition import DecompositionRule


class RXtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's RX gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose RX to RY and RZ
        return operation.gate.name == "RX"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        lambda_ = operation.params[0]
        indices = operation.qubit_indices[0]

        return [
            RZ(np.pi / 2)(indices),
            RY(lambda_)(indices),
            RZ(-np.pi / 2)(indices),
        ]


class XtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's X gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose RX to RY and RZ
        return operation.gate.name == "X"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        indices = operation.qubit_indices[0]

        return [
            RZ(np.pi / 2)(indices),
            RY(np.pi / 2)(indices),
            RZ(-np.pi / 2)(indices),
        ]
