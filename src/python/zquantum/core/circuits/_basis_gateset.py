from abc import abstractmethod
from typing import Protocol, Sequence

from _circuit import Circuit
from _gates import GateOperation

from ..decompositions._decomposition import (
    DecompositionRule,
    decompose_operation,
    decompose_operations,
)


class BasisGateset(Protocol):
    """Interface for creating basis gateset.

    A basis gateset specifies a list of gates that ever circuit should be
    composed of. It also allows the decomposition of an arbitrary gate into
    the specified basis.
    """

    @abstractmethod
    def is_overcomplete(self) -> bool:
        """Indicates that the basis contains at least one gate that can be
        decomposed into other gates in the basis.

        Returns:
            bool: true if basis is overcomplete, false otherwise.
        """
        pass

    @abstractmethod
    def decompose_operation(self, gate_operation: GateOperation) -> Circuit:
        """Given a GateOperation, return the the decdomposition of this gate
        into the basis gateset specified by this class.

        Args:
            gate_operation : Operation to be decomposed into the basis gates.

        Returns:
            Circuit: Circuit of basis gates representing the decomposition.
        """
        pass

    @abstractmethod
    def decompose_circuit(self, circuit: Circuit) -> Circuit:
        """Given a circuit return the decomposition for each gate into this
        basis gateset.

        Args:
            circuit : Circuit to be decomposed into this basis gateset

        Returns:
            Circuit: Circuit equivalent to the input, but in the basis gateset.
        """
        pass


class RzRxCx(BasisGateset):
    def __init__(self, decomposition_rules: Sequence[DecompositionRule]) -> None:
        self.decomposition_rules = decomposition_rules

    def is_overcomplete(self) -> bool:
        return False

    def decompose_operation(self, gate_operation: GateOperation) -> Circuit:
        return Circuit(decompose_operation(gate_operation, self.decomposition_rules))

    def decompose_circuit(self, circuit: Circuit) -> Circuit:
        return Circuit(
            decompose_operations(circuit.operations, self.decomposition_rules)
        )
