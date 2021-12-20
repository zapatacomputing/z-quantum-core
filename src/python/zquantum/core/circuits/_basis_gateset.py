from abc import abstractmethod
from typing import Protocol, Sequence


from _circuit import Circuit
from _gates import GateOperation
from ..decompositions._decomposition import DecompositionRule, decompose_operation


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
    def decompose(
        self, gate_operation: GateOperation, rule: DecompositionRule
    ) -> Circuit:
        """Given a GateOperation, return the the decdomposition of this gate
        into the basis gateset specified by this class.

        Args:
            gate_operation : Operation to be decomposed into the basis gates.

        Returns:
            Circuit: Circuit of basis gates representing the decomposition.
        """
        # if rule.predicate(self, gate_operation) is False:
        #     raise ValueError("Sorry, can't decompose your bloody gate, go take a hike.")
        pass


class RzRxCx(BasisGateset):
    def __init__(self, decomposition_rules: Sequence[DecompositionRule]) -> None:
        self.decomposition_rules = decomposition_rules

    def is_overcomplete(self) -> bool:
        return False
    
    def decompose(self, gate_operation: GateOperation) -> Circuit:
        return Circuit(decompose_operation(gate_operation, self.decomposition_rules))
    
    def decompose_circuit(self, circuit: Circuit) -> Circuit:
        return Circuit(decompose_operations(circuit.operations, self.decomposition_rules))
