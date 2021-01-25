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
    Dagger,
)


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
}


def extract_angle_from_gates_exponent(gate: cirq.EigenGate) -> Union[sympy.Expr, float]:
    if isinstance(gate.exponent, sympy.Basic):
        return gate.exponent * sympy.pi
    else:
        return gate.exponent * np.pi


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


def rotation_or_pauli_factory_from_eigengate(
    gate, orquestra_pauli_cls, orquestra_rotation_cls
):
    if gate.global_shift == 0 and gate.exponent == 1:
        return orquestra_pauli_cls
    elif gate.global_shift == -0.5:
        return lambda *qubits: orquestra_rotation_cls(
            *qubits, extract_angle_from_gates_exponent(gate)
        )
    else:
        raise NotImplementedError(f"Conversion of arbitrary {type(gate)} gate not yet supported.")


@orquestra_gate_factory_from_cirq_gate.register
def identity_gate_factory_from_cirq_identity(gate: cirq.IdentityGate):
    return I


@orquestra_gate_factory_from_cirq_gate.register
def oquestra_gate_factory_from_xpow_gate(gate: cirq.XPowGate):
    return rotation_or_pauli_factory_from_eigengate(gate, X, RX)


@orquestra_gate_factory_from_cirq_gate.register
def orquestra_gate_factory_from_ypow_gate(gate: cirq.YPowGate):
    return rotation_or_pauli_factory_from_eigengate(gate, Y, RY)


@orquestra_gate_factory_from_cirq_gate.register
def orquestra_gate_factory_from_zpow_gate(gate: cirq.ZPowGate):
    if gate.global_shift == 0 and gate.exponent == 0.25:
        return T
    return rotation_or_pauli_factory_from_eigengate(gate, Z, RZ)


@orquestra_gate_factory_from_cirq_gate.register
def orquestra_gate_factory_from_hpow_gate(gate: cirq.HPowGate):
    if gate.global_shift == 0 and gate.exponent == 1.0:
        return H
    raise NotImplementedError("Conversion for arbitrary HPowGate not implemented.")


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
