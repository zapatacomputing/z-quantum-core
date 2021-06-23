import operator
from functools import reduce, singledispatch
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import sympy

from . import _gates


def _circuit_size_by_operations(operations):
    return (
        0
        if not operations
        else max(
            qubit_index
            for operation in operations
            for qubit_index in operation.qubit_indices
        )
        + 1
    )


def _operation_uses_custom_gate(operation):
    return isinstance(operation.gate, _gates.MatrixFactoryGate) and isinstance(
        operation.gate.matrix_factory, _gates.CustomGateMatrixFactory
    )


class Circuit:
    """ZQuantum representation of a quantum circuit.

    See `help(zquantum.core.circuits)` for usage guide.
    """

    def __init__(
        self,
        operations: Optional[Iterable[_gates.GateOperation]] = None,
        n_qubits: Optional[int] = None,
    ):
        self._operations = list(operations) if operations is not None else []
        self._n_qubits = (
            n_qubits
            if n_qubits is not None
            else _circuit_size_by_operations(self._operations)
        )

    @property
    def operations(self):
        """Sequence of quantum gates to apply to qubits in this circuit."""
        return self._operations

    @property
    def n_qubits(self) -> int:
        """Number of qubits in this circuit.
        Not every qubit has to be used by a gate.
        """
        return self._n_qubits

    @property
    def free_symbols(self) -> List[sympy.Symbol]:
        """Set of all the sympy symbols used as params of gates in the circuit.
        The output list is sorted based on the order of appearance
        in `self._operations`."""
        seen_symbols = set()
        symbols_sequence = []
        for operation in self._operations:
            for symbol in operation.free_symbols:
                if symbol not in seen_symbols:
                    seen_symbols.add(symbol)
                    symbols_sequence.append(symbol)

        return symbols_sequence

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        if self.n_qubits != other.n_qubits:
            return False

        return list(self.operations) == list(other.operations)

    def __add__(self, other: Union["Circuit", _gates.GateOperation]):
        return _append_to_circuit(other, self)

    def collect_custom_gate_definitions(self) -> Iterable[_gates.CustomGateDefinition]:
        custom_gate_definitions = (
            operation.gate.matrix_factory.gate_definition
            for operation in self.operations
            if _operation_uses_custom_gate(operation)
        )
        unique_operation_dict = {}
        for gate_def in custom_gate_definitions:
            if gate_def.gate_name not in unique_operation_dict:
                unique_operation_dict[gate_def.gate_name] = gate_def
            elif unique_operation_dict[gate_def.gate_name] != gate_def:
                raise ValueError(
                    "Different gate definitions with the same name exist: "
                    f"{gate_def.gate_name}."
                )
        return sorted(
            unique_operation_dict.values(), key=operator.attrgetter("gate_name")
        )

    def to_unitary(self) -> Union[np.ndarray, sympy.Matrix]:
        """Create a unitary matrix describing Circuit's action.

        For performance reasons, this method will construct numpy matrix if circuit does
        not have free parameters, and a sympy matrix otherwise.
        """
        # The `reversed` iterator reflects the fact the matrices are multiplied
        # when composing linear operations (i.e. first operation is the rightmost).
        lifted_matrices = [
            op.lifted_matrix(self.n_qubits) for op in reversed(self.operations)
        ]
        return reduce(operator.matmul, lifted_matrices)

    def bind(self, symbols_map: Dict[sympy.Symbol, Any]):
        """Create a copy of the current circuit with the parameters of each gate bound
        to the values provided in the input symbols map.

        Args:
            symbols_map: A map of the symbols/gate parameters to new values
        """
        return type(self)(
            operations=[op.bind(symbols_map) for op in self.operations],
            n_qubits=self.n_qubits,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}"
            f"(operations=[{', '.join(map(str, self.operations))}], "
            f"n_qubits={self.n_qubits})"
        )


@singledispatch
def _append_to_circuit(other, circuit: Circuit):
    raise NotImplementedError()


@_append_to_circuit.register
def _append_operation(other: _gates.GateOperation, circuit: Circuit):
    n_qubits_by_operation = max(other.qubit_indices) + 1
    return type(circuit)(
        operations=[*circuit.operations, other],
        n_qubits=max(circuit.n_qubits, n_qubits_by_operation),
    )


@_append_to_circuit.register
def _append_circuit(other: Circuit, circuit: Circuit):
    return type(circuit)(
        operations=[*circuit.operations, *other.operations],
        n_qubits=max(circuit.n_qubits, other.n_qubits),
    )
