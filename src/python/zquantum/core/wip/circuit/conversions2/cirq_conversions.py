"""Orquestra <-> Cirq conversions."""
from functools import singledispatch
from operator import attrgetter
from typing import Union, Callable, Type

import cirq
import numpy as np
import sympy

from .. import _builtin_gates as bg
from .. import _gates as g


Parameter = Union[sympy.Expr, float]
RotationGateFactory = Callable[[Parameter], cirq.EigenGate]


def angle_to_exponent(angle: Parameter) -> Parameter:
    """Convert exponent from Cirq gate to angle usable in rotation gates..

    Args:
        angle: Exponent to be converted.
    Returns:
        angle divided by pi.
    Notes:
        Scaling of the angle preserves its "type", i.e. numerical angles
        are scaled by numerical approximation of pi, but symbolic ones
        are scaled by `sympy.pi`.

        This transformation might be viewed as the change of units from
        radians to pi * radians.
    """
    return angle / (sympy.pi if isinstance(angle, sympy.Expr) else np.pi)


def make_rotation_factory(
    eigengate_cls: Type[cirq.EigenGate], global_shift: float = 0
) -> RotationGateFactory:
    """Construct a factory for rotation gate based on given EigenGate subclass.

    This function might be thought of as a partial which freezes global_shift
    parameter but also scales the exponent parameter of eigengate_cls initializer.

    Args:
        eigengate_cls: EigenGate subclass, e.g. ZPowGate, XXPowGate.
        global_shift: Determines phase of the rotation gate. Check Cirq docs
            for explanation.
    Returns:
        A function that maps angle to EigenGate instance with given global shift
        and an exponent equal to angle divided by a factor of pi.
    """
    def _rotation(angle: Parameter) -> cirq.EigenGate:
        return eigengate_cls(
            global_shift=global_shift, exponent=angle_to_exponent(angle)
        )

    return _rotation


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

EIGENGATE_ROTATIONS = {
    (cirq.XPowGate, -0.5): bg.RX,
    (cirq.YPowGate, -0.5): bg.RY,
    (cirq.ZPowGate, -0.5): bg.RZ,
    (cirq.ZPowGate, 0): bg.PHASE,
    (cirq.CZPowGate, 0): bg.CPHASE,
    (cirq.XXPowGate, -0.5): bg.XX,
    (cirq.YYPowGate, -0.5): bg.YY,
    (cirq.ZZPowGate, -0.5): bg.ZZ,
    (cirq.ISwapPowGate, 0.0): bg.XY,
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
