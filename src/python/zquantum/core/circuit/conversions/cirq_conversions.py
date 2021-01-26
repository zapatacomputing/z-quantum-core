from functools import singledispatch
from typing import Union

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
    ZZ
)


def make_rotation_factory(eigengate_cls, global_shift: float=0):
    def _rotation(angle):
        return eigengate_cls(global_shift=global_shift, exponent=angle_to_exponent(angle))

    return _rotation


# Mapping between Orquestra gate classes and Cirq gates.
# Note that not all gates are included, those require special treatment.
ORQUESTRA_TO_CIRQ_MAPPING = {
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
}


EIGENGATE_SPECIAL_CASES = {
    (type(cirq.X), cirq.X.global_shift, cirq.X.exponent): X,
    (type(cirq.Y), cirq.Y.global_shift, cirq.Y.exponent): Y,
    (type(cirq.Z), cirq.Z.global_shift, cirq.Z.exponent): Z,
    (type(cirq.T), cirq.T.global_shift, cirq.T.exponent): T,
    (type(cirq.H), cirq.H.global_shift, cirq.H.exponent): H,
    (type(cirq.CNOT), cirq.CNOT.global_shift, cirq.CNOT.exponent): CNOT,
    (type(cirq.CZ), cirq.CZ.global_shift, cirq.CZ.exponent): CZ,
    (type(cirq.SWAP), cirq.SWAP.global_shift, cirq.SWAP.exponent): SWAP
}


EIGENGATE_ROTATIONS = {
    (cirq.XPowGate, -0.5): RX,
    (cirq.YPowGate, -0.5): RY,
    (cirq.ZPowGate, -0.5): RZ,
    (cirq.ZPowGate, 0): PHASE,
    (cirq.CZPowGate, 0): CPHASE,
    (cirq.XXPowGate, -0.5): XX,
    (cirq.YYPowGate, -0.5): YY,
    (cirq.ZZPowGate, -0.5): ZZ
}


def extract_angle_from_gates_exponent(gate: cirq.EigenGate) -> Union[sympy.Expr, float]:
    if isinstance(gate.exponent, sympy.Basic):
        return gate.exponent * sympy.pi
    else:
        return gate.exponent * np.pi


def angle_to_exponent(angle: Union[sympy.Expr, float]):
    if isinstance(angle, sympy.Expr):
        return angle / sympy.pi
    else:
        return angle / np.pi


@singledispatch
def convert_to_cirq(obj):
    raise NotImplementedError(f"Cannot convert {obj} to cirq object.")


@convert_to_cirq.register
def convert_orquestra_gate_to_cirq(gate: Gate):
    try:
        cirq_gate = ORQUESTRA_TO_CIRQ_MAPPING[type(gate)]
        if gate.params:
            cirq_gate = cirq_gate(*gate.params)
        return cirq_gate(*(cirq.LineQubit(qubit) for qubit in gate.qubits))
    except KeyError:
        raise NotImplementedError(
            f"Cannot convert Orquestra gate {gate} to cirq. This is probably a bug, "
            "please reach out to Orquestra support."
        )


@singledispatch
def convert_from_cirq(obj):
    raise NotImplementedError(
        f"Conversion from cirq to Orquestra not supported for {obj}."
    )


@singledispatch
def orquestra_gate_factory_from_cirq_gate(gate: cirq.Gate):
    raise NotImplementedError(f"Don't know Orquestra factory for gate: {gate}.")


@orquestra_gate_factory_from_cirq_gate.register
def identity_gate_factory_from_cirq_identity(_gate: cirq.IdentityGate):
    return I


@orquestra_gate_factory_from_cirq_gate.register
def oquestra_gate_factory_from_xpow_gate(gate: cirq.EigenGate):
    key = (type(gate), gate.global_shift, gate.exponent)
    if key in EIGENGATE_SPECIAL_CASES:
        return EIGENGATE_SPECIAL_CASES[key]
    elif key[0:2] in EIGENGATE_ROTATIONS:
        return lambda *qubits: EIGENGATE_ROTATIONS[key[0:2]](
            *qubits, extract_angle_from_gates_exponent(gate)
        )
    else:
        raise NotImplementedError(
            f"Conversion of arbitrary {type(gate).__name__} gate not supported yet."
        )


@convert_from_cirq.register
def convert_cirq_gate_operation_to_orquestra_gate(ops: cirq.ops.GateOperation):
    if not all(isinstance(qubit, cirq.LineQubit) for qubit in ops.qubits):
        raise NotImplementedError(
            "Currently conversions from cirq to Orquestra is supported only for "
            "gate operations with LineQubits."
        )

    orquestra_gate_factory = orquestra_gate_factory_from_cirq_gate(ops.gate)
    orquestra_qubits = [qubit.x for qubit in ops.qubits]

    return orquestra_gate_factory(*orquestra_qubits)
