from typing import Iterable

import numpy as np
from zquantum.core.circuits._builtin_gates import GPHASE, RY, RZ
from zquantum.core.circuits._gates import GateOperation
from zquantum.core.decompositions._decomposition import DecompositionRule


class PHASEtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's PHASE gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose PHASE to RY and RZ
        return operation.gate.name == "PHASE"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        alpha = operation.params[0]
        indices = operation.qubit_indices[0]

        return [
            RZ(alpha)(indices),
            GPHASE(alpha / 2)(indices),
        ]


class RXtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's RX gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose RX to RY and RZ
        return operation.gate.name == "RX"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        theta = operation.params[0]
        indices = operation.qubit_indices[0]

        return [
            RZ(-np.pi/2)(indices),
            RY(theta)(indices),
            RZ(np.pi/2)(indices),
        ]


class U3toRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's U3 gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose U3 to RY and RZ
        return operation.gate.name == "U3"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        phi = operation.params[0]
        theta = operation.params[1]
        lambda_ = operation.params[2]
        indices = operation.qubit_indices[0]

        return [
            RZ(phi)(indices),
            RY(theta)(indices),
            RZ(lambda_)(indices),
            GPHASE((phi + lambda_) / 2)(indices),
        ]


class ItoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's I gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose I to RY and RZ
        return operation.gate.name == "I"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        indices = operation.qubit_indices[0]

        return [
            RZ(0)(indices),
            RY(0)(indices),
            RZ(0)(indices),
        ]


class XtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's X gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose X to RY and RZ
        return operation.gate.name == "X"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        indices = operation.qubit_indices[0]

        return [
            GPHASE(np.pi/2)(indices),
            RZ(-np.pi/2)(indices),
            RY(np.pi)(indices),
            RZ(np.pi/2)(indices),
        ]


class YtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's Y gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose Y to RY and RZ
        return operation.gate.name == "Y"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        indices = operation.qubit_indices[0]

        return [
            GPHASE(np.pi/2)(indices),
            RY(np.pi)(indices),
        ]


class ZtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's Z gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose Z to RY and RZ
        return operation.gate.name == "Z"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        indices = operation.qubit_indices[0]

        return [
            GPHASE(np.pi/2)(indices),
            RZ(np.pi)(indices),
        ]


class HtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's H gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose H to RY and RZ
        return operation.gate.name == "H"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        indices = operation.qubit_indices[0]

        return [
            RY(np.pi/2)(indices),
            GPHASE(np.pi/2)(indices),
            RZ(np.pi)(indices),
        ]


class StoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's S gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose S to RY and RZ
        return operation.gate.name == "S"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        indices = operation.qubit_indices[0]

        return [
            RZ(np.pi / 2)(indices),
            GPHASE(np.pi / 4)(indices),
        ]


class TtoRZRY(DecompositionRule[GateOperation]):
    """Decomposition of ZQuantum's T gate in the RZRYCNOT gateset."""

    def predicate(self, operation: GateOperation) -> bool:
        # Only decompose T to RY and RZ
        return operation.gate.name == "T"

    def production(self, operation: GateOperation) -> Iterable[GateOperation]:
        indices = operation.qubit_indices[0]

        return [
            RZ(np.pi / 4)(indices),
            GPHASE(np.pi / 8)(indices),
        ]
