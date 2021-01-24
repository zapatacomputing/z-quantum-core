import re
from functools import singledispatch
from typing import Sequence

import cirq
import numpy as np

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


def extract_angle_from_gates_exponent(gate: cirq.EigenGate):
    return gate.exponent * np.pi,


# Map of cirq gate name to Orquestra's gate class.
CIRQ_TO_ORQUESTRA_CLS_MAPPING = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "I": I,
    "H": H,
    "T": T,
    "Rx": RX,
    "Ry": RY,
    "Rz": RZ
}


CIRQ_TO_ORQUESTRA_PARAMETER_CONVERTER = {
    "Rx": extract_angle_from_gates_exponent,
    "Ry": extract_angle_from_gates_exponent,
    "Rz": extract_angle_from_gates_exponent
}


def parse_gate_name_from_cirq_gate(gate: cirq.Gate) -> str:
    """Parse name of the gate given Cirq object that represents it.

    Args:
        gate: Cirq gate whose name should be parsed.
    Returns:
        Name of `gate` as parsed from its str representation.

    """
    pattern = "[A-Za-z]+"
    match = re.match(pattern, str(gate))
    return match.group(0)


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


@convert_from_cirq.register
def convert_cirq_gate_operation_to_orquestra_gate(ops: cirq.ops.GateOperation):
    if not all(isinstance(qubit, cirq.LineQubit) for qubit in ops.qubits):
        raise NotImplementedError(
            "Currently conversions from cirq to Orquestra is supported only for "
            "gate operations with LineQubits."
        )

    cirq_gate_name = parse_gate_name_from_cirq_gate(ops.gate)
    if cirq_gate_name in CIRQ_TO_ORQUESTRA_PARAMETER_CONVERTER:
        orquestra_params = CIRQ_TO_ORQUESTRA_PARAMETER_CONVERTER[cirq_gate_name](ops.gate)
    else:
        orquestra_params = ()
    orquestra_qubits = [qubit.x for qubit in ops.qubits]

    return CIRQ_TO_ORQUESTRA_CLS_MAPPING[cirq_gate_name](*orquestra_qubits, *orquestra_params)
