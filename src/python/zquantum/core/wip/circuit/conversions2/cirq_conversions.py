"""Orquestra <-> Cirq conversions."""
from functools import singledispatch
from operator import attrgetter
import cirq

from .. import _builtin_gates as bg
from .. import _gates as g


ORQUESTRA_BUILTIN_GATE_NAME_TO_CIRQ_GATE = {
    "X": cirq.X,
    "Y": cirq.Y,
    "Z": cirq.Z,
    "I": cirq.I,
    "H": cirq.H,
    "T": cirq.T,
    "CNOT": cirq.CNOT,
    "CZ": cirq.CZ,
    "SWAP": cirq.SWAP
}


EIGENGATE_SPECIAL_CASES = {
    (type(cirq.X), cirq.X.global_shift, cirq.X.exponent): bg.X,
    (type(cirq.Y), cirq.Y.global_shift, cirq.Y.exponent): bg.Y,
    (type(cirq.Z), cirq.Z.global_shift, cirq.Z.exponent): bg.Z,
    (type(cirq.T), cirq.T.global_shift, cirq.T.exponent): bg.T,
    (type(cirq.H), cirq.H.global_shift, cirq.H.exponent): bg.H,
    (type(cirq.CNOT), cirq.CNOT.global_shift, cirq.CNOT.exponent): bg.CNOT,
    (type(cirq.CZ), cirq.CZ.global_shift, cirq.CZ.exponent): bg.CZ,
    (type(cirq.SWAP), cirq.SWAP.global_shift, cirq.SWAP.exponent): bg.SWAP,
}

qubit_index = attrgetter("x")


@singledispatch
def convert_to_cirq(obj):
    raise NotImplementedError(f"{obj} not convertible to Cirq object.")


@convert_to_cirq.register
def convert_matrix_factory_gate_to_cirq(gate: g.MatrixFactoryGate) -> cirq.Gate:
    try:
        return ORQUESTRA_BUILTIN_GATE_NAME_TO_CIRQ_GATE[gate.name]
    except KeyError:
        raise NotImplementedError(f"Gate {gate} not convertible to Cirq.")


@convert_to_cirq.register
def convert_gate_operation_to_cirq(operation: g.GateOperation) -> cirq.GateOperation:
    return convert_to_cirq(operation.gate)(*map(cirq.LineQubit, operation.qubit_indices))


@singledispatch
def convert_from_cirq(obj):
    raise NotImplementedError(f"{obj} not convertible to Orquestra object.")


@convert_from_cirq.register
def convert_eigengate_to_orquestra_gate(eigengate: cirq.EigenGate) -> g.Gate:
    key = (type(eigengate), eigengate.global_shift, eigengate.exponent)
    if key in EIGENGATE_SPECIAL_CASES:
        return EIGENGATE_SPECIAL_CASES[key]
    else:
        raise NotImplementedError(f"Gate {eigengate} not convertible to Orquestra object.")


@convert_from_cirq.register
def convert_cirq_identity_gate_to_orquestra_gate(identity_gate: cirq.IdentityGate) -> g.Gate:
    return bg.I


@convert_from_cirq.register
def convert_gate_operation_to_orquestra(operation: cirq.GateOperation) -> g.GateOperation:
    if not all(isinstance(qubit, cirq.LineQubit) for qubit in operation.qubits):
        raise NotImplementedError(
            f"Failed to convert {operation}. Grid qubits are not yet supported."
        )

    return convert_from_cirq(operation.gate)(*map(qubit_index, operation.qubits))
