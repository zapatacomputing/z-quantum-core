"""Class hierarchy for base gates."""
import math
from dataclasses import dataclass
import typing as t
from typing_extensions import Protocol, abstractproperty
from functools import singledispatch

import sympy


@dataclass(frozen=True)
class GateApplication:
    gate: "Gate"
    qubit_indices: t.Iterable[int]

    def propagate(self, wave_function):
        # lift gate matrix for the whole wave function
        ...


# TODO: figure out what should be the concrete type for the Wave Function.
# The chosen type should fit nicely with the simulator backends that we indend to use.
# It doesn't matter for quantum hardware backends, because we only support transpiling
# circuits composed of well-defined quantum gates to a quantum computer.
#
# WF is a vector containing 2^(n_qubits) complex numbers that describes a quantum state,
# before a quantum collapse.
WaveFunction = t.Any


@dataclass(frozen=True)
class OpaqueOperation:
    transformation: t.Callable[[WaveFunction], WaveFunction]
    qubit_indices: t.Iterable[int]

    def propagate(self, wave_function):
        # can't be done because wave functions don't work that way
        # superposition and entanglement, yo
        # return self.transformation(wave_function[self.qubit_indices])
        return self.transformation(wave_function)


class QuantumOperation(Protocol):
    @abstractproperty
    def qubit_indices(self):
        ...

    def propagate(self, wave_function: WaveFunction) -> WaveFunction:
        """Allows running on a simulator backend or in the REPL"""
        ...


@dataclass(frozen=True)
class Gate:
    """Quantum gate defined with a matrix.

    Args:
        name: Name of this gate. Implementers of new gates should make sure that the names are
            unique.
        matrix: Unitary matrix defining action of this gate.
    """

    name: str
    matrix: sympy.Matrix

    def __call__(self, *qubit_indices) -> "GateApplication":
        return GateApplication(self, qubit_indices)


def _n_qubits_for_matrix(matrix_shape):
    n_qubits = math.floor(math.log2(matrix_shape[0]))
    if 2 ** n_qubits != matrix_shape[0] or 2 ** n_qubits != matrix_shape[1]:
        raise ValueError("Gate's matrix has to be square with dimension 2^N")

    return n_qubits


def make_parametric_gate_factory(
    name: str,
    matrix_factory
):
    def _gate_factory(*params):
        return Gate(
            name=name,
            matrix=matrix_factory(*params)
        )

    return _gate_factory


@dataclass(frozen=True)
class Circuit:
    operations: t.Iterable[QuantumOperation]
    n_qubits: int

    def __add__(self, other: "Circuit"):
        return _append_to_circuit(other, self)


@singledispatch
def _append_to_circuit(other, circuit: Circuit):
    raise NotImplementedError()


@_append_to_circuit.register
def _append_gate(other_gate: Gate, circuit: Circuit):
    n_qubits_by_gate = max(other_gate.qubits) + 1
    return type(circuit)(operations=[*circuit.operations, other_gate], n_qubits=max(circuit.n_qubits, n_qubits_by_gate))


@_append_to_circuit.register
def _append_circuit(other_circuit: Circuit, circuit: Circuit):
    return type(circuit)(gates=[*circuit.gates, *other_circuit.gates], n_qubits=max(circuit.n_qubits, other_circuit.n_qubits))
