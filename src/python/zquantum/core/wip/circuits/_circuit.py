from functools import singledispatch, reduce
import operator
from typing import Union, Dict, Optional, Iterable, Any

import sympy

from . import _gates, MatrixFactoryGate
from ._gates import CustomGateMatrixFactory


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


def _bind_operation(op: _gates.GateOperation, symbols_map) -> _gates.GateOperation:
    return op.gate.bind(symbols_map)(*op.qubit_indices)


def _operation_uses_custom_gate(operation):
    return isinstance(operation.gate, MatrixFactoryGate) and isinstance(
        operation.gate.matrix_factory, CustomGateMatrixFactory
    )


class Circuit:
    """ZQuantum representation of a quantum circuit."""

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
    def n_qubits(self):
        """Number of qubits in this circuit.
        Not every qubit has to be used by a gate.
        """
        return self._n_qubits

    @property
    def free_symbols(self):
        """Set of all the sympy symbols used as params of gates in the circuit."""
        return reduce(
            set.union,
            (operation.gate.free_symbols for operation in self._operations),
            set(),
        )

    def __eq__(self, other: "Circuit"):
        if not isinstance(other, type(self)):
            return False

        if self.n_qubits != other.n_qubits:
            return False

        if list(self.operations) != list(other.operations):
            return False

        return True

    def __add__(self, other: Union["Circuit"]):
        return _append_to_circuit(other, self)

    def collect_custom_gate_definitions(self):
        custom_gate_definiions = (
            operation.gate.matrix_factory.gate_definition
            for operation in self.operations
            if _operation_uses_custom_gate(operation)
        )
        unique_operation_dict = {}
        for gate_def in custom_gate_definiions:
            if gate_def.gate_name not in unique_operation_dict:
                unique_operation_dict[gate_def.gate_name] = gate_def
            elif unique_operation_dict[gate_def.gate_name] != gate_def:
                raise ValueError(
                    f"Different gate definitions with the same name exist: {gate_def.gate_name}."
                )
        return sorted(
            unique_operation_dict.values(), key=operator.attrgetter("gate_name")
        )

    def bind(self, symbols_map: Dict[sympy.Symbol, Any]):
        """Create a copy of the current circuit with the parameters of each gate bound to
        the values provided in the input symbols map

        Args:
            symbols_map: A map of the symbols/gate parameters to new values
        """
        return type(self)(
            operations=[_bind_operation(op, symbols_map) for op in self.operations],
            n_qubits=self.n_qubits,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}"
            f"(operations=[{', '.join(map(str, self.operations))}], "
            f"n_qubits={self.n_qubits}, "
            f"custom_gate_definitions={self.custom_gate_definitions})"
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
