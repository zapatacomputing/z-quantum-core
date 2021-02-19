"""Utilities for converting gates and circuits to and from Cirq objects."""
from functools import singledispatch
from typing import Union, Type, Callable

import cirq
import numpy as np
import sympy

from ...circuit.gates import (
    Gate,
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    PHASE,
    T,
    I,
    H,
    CZ,
    CNOT,
    CPHASE,
    SWAP,
    XX,
    YY,
    ZZ,
    XY,
)

Param = Union[sympy.Expr, float]
RotationGateFactory = Callable[[Param], cirq.EigenGate]


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

    def _rotation(angle: Param) -> cirq.EigenGate:
        return eigengate_cls(
            global_shift=global_shift, exponent=angle_to_exponent(angle)
        )

    return _rotation


# Mapping between Orquestra gate classes and Cirq gates.
# Note that not all gates are included, those require special treatment.
ZQUANTUM_TO_CIRQ_MAPPING = {
    X: cirq.X,
    Y: cirq.Y,
    Z: cirq.Z,
    RX: cirq.rx,
    RY: cirq.ry,
    RZ: cirq.rz,
    T: cirq.T,
    I: cirq.I,
    H: cirq.H,
    CZ: cirq.CZ,
    CNOT: cirq.CNOT,
    SWAP: cirq.SWAP,
    PHASE: make_rotation_factory(cirq.ZPowGate),
    CPHASE: make_rotation_factory(cirq.CZPowGate),
    XX: make_rotation_factory(cirq.XXPowGate, -0.5),
    YY: make_rotation_factory(cirq.YYPowGate, -0.5),
    ZZ: make_rotation_factory(cirq.ZZPowGate, -0.5),
    XY: make_rotation_factory(cirq.ISwapPowGate, 0.0),
}


EIGENGATE_SPECIAL_CASES = {
    (type(cirq.X), cirq.X.global_shift, cirq.X.exponent): X,
    (type(cirq.Y), cirq.Y.global_shift, cirq.Y.exponent): Y,
    (type(cirq.Z), cirq.Z.global_shift, cirq.Z.exponent): Z,
    (type(cirq.T), cirq.T.global_shift, cirq.T.exponent): T,
    (type(cirq.H), cirq.H.global_shift, cirq.H.exponent): H,
    (type(cirq.CNOT), cirq.CNOT.global_shift, cirq.CNOT.exponent): CNOT,
    (type(cirq.CZ), cirq.CZ.global_shift, cirq.CZ.exponent): CZ,
    (type(cirq.SWAP), cirq.SWAP.global_shift, cirq.SWAP.exponent): SWAP,
}


EIGENGATE_ROTATIONS = {
    (cirq.XPowGate, -0.5): RX,
    (cirq.YPowGate, -0.5): RY,
    (cirq.ZPowGate, -0.5): RZ,
    (cirq.ZPowGate, 0): PHASE,
    (cirq.CZPowGate, 0): CPHASE,
    (cirq.XXPowGate, -0.5): XX,
    (cirq.YYPowGate, -0.5): YY,
    (cirq.ZZPowGate, -0.5): ZZ,
    (cirq.ISwapPowGate, 0.0): XY,
}


def exponent_to_angle(exponent: Param) -> Param:
    """Convert exponent from Cirq gate to angle usable in rotation gates..

    Args:
        exponent: Exponent to be converted.
    Returns:
        exponent multiplied by pi.
    Notes:
        Scaling of the exponent preserves its "type", i.e. numerical exponents
        are scaled by numerical approximation of pi, but symbolic ones
        are scaled by sympy.pi
    """
    return exponent * (sympy.pi if isinstance(exponent, sympy.Expr) else np.pi)


def angle_to_exponent(angle: Param) -> Param:
    """Convert exponent from Cirq gate to angle usable in rotation gates..

    Args:
        angle: Exponent to be converted.
    Returns:
        angle divided by pi.
    Notes:
        Scaling of the angle preserves its "type", i.e. numerical angles
        are scaled by numerical approximation of pi, but symbolic ones
        are scaled by sympy.pi
    """
    return angle / (sympy.pi if isinstance(angle, sympy.Expr) else np.pi)


@singledispatch
def convert_to_cirq(obj):
    """Convert native Orquestra object to its closest Cirq counterpart.

    TODO: add longer description once all conversions are done.
    """
    raise NotImplementedError(f"Cannot convert {obj} to cirq object.")


@convert_to_cirq.register
def convert_orquestra_gate_to_cirq(gate: Gate) -> cirq.GateOperation:
    """Convert native Orquestra get to its Cirq counterpart."""
    try:
        cirq_gate = ZQUANTUM_TO_CIRQ_MAPPING[type(gate)]
        if gate.params:
            cirq_gate = cirq_gate(*gate.params)
        return cirq_gate(*(cirq.LineQubit(qubit) for qubit in gate.qubits))
    except KeyError:
        raise NotImplementedError(
            f"Cannot convert Orquestra gate {gate} to cirq. This is probably a bug, "
            "please reach out to the Orquestra support."
        )


@singledispatch
def convert_from_cirq(obj):
    """Convert Cirq object to its closes Orquestra counterpart.

    TODO: add longer description once all conversions are done.
    """
    raise NotImplementedError(
        f"Conversion from cirq to Orquestra not supported for {obj}."
    )


@singledispatch
def orquestra_gate_factory_from_cirq_gate(gate: cirq.Gate) -> Callable[..., Gate]:
    """Create a function that constructs orquestra Gate based on Cirq Gate.

    Args:
          gate: Cirq gate to base the factory on.
    Returns:
          function of the form f(*qubits) -> Gate. For most variants of this
          function those will be just respective classes initializers.
    """
    raise NotImplementedError(f"Don't know Orquestra factory for gate: {gate}.")


@orquestra_gate_factory_from_cirq_gate.register
def identity_gate_factory_from_cirq_identity(_gate: cirq.IdentityGate) -> Type[I]:
    return I


@orquestra_gate_factory_from_cirq_gate.register
def orquestra_gate_factory_from_eigengate(gate: cirq.EigenGate) -> Callable[..., Gate]:
    key = (type(gate), gate.global_shift, gate.exponent)
    if key in EIGENGATE_SPECIAL_CASES:
        return EIGENGATE_SPECIAL_CASES[key]
    elif key[0:2] in EIGENGATE_ROTATIONS:
        return lambda *qubits: EIGENGATE_ROTATIONS[key[0:2]](
            *qubits, angle=exponent_to_angle(gate.exponent)
        )
    else:
        raise NotImplementedError(
            f"Conversion of arbitrary {type(gate).__name__} gate not supported yet."
        )


@convert_from_cirq.register
def convert_cirq_gate_operation_to_orquestra_gate(
    operation: cirq.GateOperation,
) -> Gate:
    """Convert Cirq GateOperation to native Orquestra Gate.

    Note that Cirq distinguishes concept of GateOperation and Gate. The correspondence
    between terminology in Cirq is as follows:

    Cirq Gate <-> Orquestra Gate subclass
    Cirq GateOperation <-> A concrete instance of some Orquestra Gate,

    which is why this function accepts GateOperation instead of Gate.

    Args:
        operation: operation to be converted.
    Returns:
        Native Orquestra Gate.
    """
    if not all(isinstance(qubit, cirq.LineQubit) for qubit in operation.qubits):
        raise NotImplementedError(
            "Currently conversions from cirq to Orquestra is supported only for "
            "gate operations with LineQubits."
        )

    orquestra_gate_factory = orquestra_gate_factory_from_cirq_gate(operation.gate)
    orquestra_qubits = [qubit.x for qubit in operation.qubits]

    return orquestra_gate_factory(*orquestra_qubits)
