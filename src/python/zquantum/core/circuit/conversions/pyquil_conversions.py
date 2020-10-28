"""Utilities for converting gates and circuits to and from Pyquil objects."""
from functools import singledispatch
from typing import Union
import numpy as np
import pyquil
import pyquil.gates
from ...circuit import Gate, ControlledGate
from ...circuit.gates import X, Y, Z, RX, RY, RZ, PHASE, T, I, H, Dagger, CZ, CNOT, CPHASE, SWAP

SINGLE_QUBIT_NONPARAMETRIC_GATES = {
    X: pyquil.gates.X,
    Y: pyquil.gates.Y,
    Z: pyquil.gates.Z,
    I: pyquil.gates.I,
    T: pyquil.gates.T,
    H: pyquil.gates.H
}


ROTATION_GATES = {
    RX: pyquil.gates.RX, RY: pyquil.gates.RY, RZ: pyquil.gates.RZ, PHASE: pyquil.gates.PHASE
}


TWO_QUBIT_CONTROLLED_GATES = {
    CZ: pyquil.gates.CZ, CNOT: pyquil.gates.CNOT, SWAP: pyquil.gates.SWAP
}


@singledispatch
def convert_to_pyquil(obj):
    raise NotImplementedError(f"Cannot convert {obj} to PyQuil object.")


@convert_to_pyquil.register
def convert_gate_to_pyquil(gate: Gate, program: Optional[pyquil.Program] = None) -> pyquil.gates.Gate:
    if gate.symbolic_params:
        raise NotImplementedError(f"Cannot convert gate with symbolic params to PyQuil object.")
    return _convert_gate_to_pyquil(gate)


@singledispatch
def _convert_gate_to_pyquil(gate) -> pyquil.gates.Gate:
    raise NotImplementedError(f"Cannot convert gate {gate} to PyQUil.")


@_convert_gate_to_pyquil.register(X)
@_convert_gate_to_pyquil.register(Y)
@_convert_gate_to_pyquil.register(Z)
@_convert_gate_to_pyquil.register(I)
@_convert_gate_to_pyquil.register(T)
@_convert_gate_to_pyquil.register(H)
def convert_single_qubit_nonparametric_gate_to_pyquil(
    gate: Union[X, Y, Z]
) -> pyquil.gates.Gate:
    return SINGLE_QUBIT_NONPARAMETRIC_GATES[type(gate)](gate.qubit)


@_convert_gate_to_pyquil.register(RX)
@_convert_gate_to_pyquil.register(RY)
@_convert_gate_to_pyquil.register(RZ)
@_convert_gate_to_pyquil.register(PHASE)
def convert_single_qubit_rotation_gate_to_pyquil(
    gate: Union[RX, RY, RZ, PHASE]
) -> pyquil.gates.Gate:
    return ROTATION_GATES[type(gate)](gate.angle, gate.qubit)


@_convert_gate_to_pyquil.register(CNOT)
@_convert_gate_to_pyquil.register(CZ)
@_convert_gate_to_pyquil.register(SWAP)
def convert_two_qubit_nonparametric_gate_to_pyquil(
    gate: Union[CZ]
) -> pyquil.gates.Gate:
    return TWO_QUBIT_CONTROLLED_GATES[type(gate)](*gate.qubits)


@_convert_gate_to_pyquil.register(CPHASE)
def convert_CPHASE_to_pyquil(gate: CPHASE) -> pyquil.gates.Gate:
    return pyquil.gates.CPHASE(gate.angle, *gate.qubits)


@_convert_gate_to_pyquil.register(SWAP)
def convert_SWAP_gate_to_pyquil(gate: SWAP) -> pyquil.gates.Gate:
    return pyquil.gates.SWAP(*gate.qubits)


@_convert_gate_to_pyquil.register(ControlledGate)
def convert_controlled_gate_to_pyquil(gate: ControlledGate) -> pyquil.gates.Gate:
    return convert_to_pyquil(gate.target_gate).controlled(gate.control)


@_convert_gate_to_pyquil.register(Dagger)
def convert_dagger_to_pyquil(gate: Dagger) -> pyquil.gates.Gate:
    return convert_to_pyquil(gate.gate).dagger()
