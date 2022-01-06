from abc import abstractmethod
from typing import Optional, Sequence, Tuple

from typing_extensions import Protocol
from zquantum.core.circuits._circuit import Circuit
from zquantum.core.circuits._gates import GateOperation
from zquantum.core.decompositions._decomposition import (
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


class RZRYCNOT(BasisGateset):
    def __init__(self, decomposition_rules: Sequence[DecompositionRule]) -> None:
        self.basis_gates = ("RZ", "RY", "CNOT")
        self.decomposition_rules = decomposition_rules

    def is_overcomplete(self) -> bool:
        return False

    def decompose_operation(self, gate_operation: GateOperation) -> Circuit:
        decomposed_circuit = Circuit(
            decompose_operation(gate_operation, self.decomposition_rules)
        )
        invalid_operation = _is_valid_decomposition(
            decomposed_circuit, self.basis_gates
        )
        if invalid_operation is not None:
            raise RuntimeError(
                "Failed to decompose the operation"
                f"'{invalid_operation}' into the basis gateset '{self.basis_gates}'"
            )

        return decomposed_circuit

    def decompose_circuit(self, circuit: Circuit) -> Circuit:
        decomposed_circuit = Circuit(
            decompose_operations(circuit.operations, self.decomposition_rules)
        )
        invalid_operation = _is_valid_decomposition(
            decomposed_circuit, self.basis_gates
        )
        if invalid_operation is not None:
            raise RuntimeError(
                "Failed to decompose the operation"
                f"'{invalid_operation}' into the basis gateset '{self.basis_gates}'"
            )

        return decomposed_circuit


def _is_valid_decomposition(
    circuit: Circuit, basis_gates: Tuple[str, ...]
) -> Optional[GateOperation]:
    """Checks if circuit is composed entirely of the gates provided in basis_gates.

    Args:
        circuit : Circuit to be checked if if is composed of basis gates.
        basis_gates : Names of the gates which we want to check circuit is composed of.

    Returns:
        If an operation in circuit is not in the basis_gates, return the operation,
        otherwise return None.
    """
    for operation in circuit.operations:
        if operation.gate.name not in basis_gates:
            return operation

    return None
