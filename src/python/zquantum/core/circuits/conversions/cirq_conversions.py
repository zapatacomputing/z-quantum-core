"""Zquantum <-> Cirq conversions."""
import hashlib
from dataclasses import dataclass
from functools import singledispatch
from itertools import chain
from operator import attrgetter
from typing import Callable, Dict, Type, Union, overload

import cirq
import numpy as np
import sympy

from .. import _builtin_gates, _circuit, _gates

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
        are scaled by `sympy.pi`. Note that in case of sympy numbers,
        the results is a native float.

        This transformation might be viewed as the change of units from
        radians to pi * radians.
    """
    if isinstance(angle, sympy.Expr) and angle.is_constant():
        try:
            angle = float(angle)
        # This broad except is intentional. I am not sure if it is always
        # possible to convert constant sympy.Expr to float, therefore
        # we allow graceful recovery in case the conversion failed.
        except Exception:
            pass
    return angle / (sympy.pi if isinstance(angle, sympy.Expr) else np.pi)


def exponent_to_angle(exponent: Parameter) -> Parameter:
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


def _cirq_u3_factory(*args):
    return cirq.circuits.qasm_output.QasmUGate(*map(angle_to_exponent, args))


ZQUANTUM_BUILTIN_GATE_NAME_TO_CIRQ_GATE: Dict[str, Callable] = {
    "X": cirq.X,
    "Y": cirq.Y,
    "Z": cirq.Z,
    "I": cirq.I,
    "H": cirq.H,
    "S": cirq.S,
    "T": cirq.T,
    "RX": cirq.rx,
    "RY": cirq.ry,
    "RZ": cirq.rz,
    "RH": make_rotation_factory(cirq.HPowGate, 0.0),
    "PHASE": make_rotation_factory(cirq.ZPowGate),
    "CNOT": cirq.CNOT,
    "CZ": cirq.CZ,
    "SWAP": cirq.SWAP,
    "ISWAP": cirq.ISWAP,
    "CPHASE": cirq.cphase,
    "XX": make_rotation_factory(cirq.XXPowGate, -0.5),
    "YY": make_rotation_factory(cirq.YYPowGate, -0.5),
    "ZZ": make_rotation_factory(cirq.ZZPowGate, -0.5),
    "XY": make_rotation_factory(cirq.ISwapPowGate, 0.0),
    "U3": _cirq_u3_factory,
}


EIGENGATE_SPECIAL_CASES = {
    (type(cirq.X), cirq.X.global_shift, cirq.X.exponent): _builtin_gates.X,
    (type(cirq.Y), cirq.Y.global_shift, cirq.Y.exponent): _builtin_gates.Y,
    (type(cirq.Z), cirq.Z.global_shift, cirq.Z.exponent): _builtin_gates.Z,
    (type(cirq.S), cirq.S.global_shift, cirq.S.exponent): _builtin_gates.S,
    (type(cirq.T), cirq.T.global_shift, cirq.T.exponent): _builtin_gates.T,
    (type(cirq.H), cirq.H.global_shift, cirq.H.exponent): _builtin_gates.H,
    (type(cirq.CNOT), cirq.CNOT.global_shift, cirq.CNOT.exponent): _builtin_gates.CNOT,
    (type(cirq.CZ), cirq.CZ.global_shift, cirq.CZ.exponent): _builtin_gates.CZ,
    (type(cirq.SWAP), cirq.SWAP.global_shift, cirq.SWAP.exponent): _builtin_gates.SWAP,
    (
        type(cirq.ISWAP),
        cirq.ISWAP.global_shift,
        cirq.ISWAP.exponent,
    ): _builtin_gates.ISWAP,
    (
        cirq.ops.common_gates.XPowGate,
        cirq.X.global_shift,
        cirq.X.exponent,
    ): _builtin_gates.X,
    (
        cirq.ops.common_gates.YPowGate,
        cirq.Y.global_shift,
        cirq.Y.exponent,
    ): _builtin_gates.Y,
    (
        cirq.ops.common_gates.ZPowGate,
        cirq.Z.global_shift,
        cirq.Z.exponent,
    ): _builtin_gates.Z,
}

EIGENGATE_ROTATIONS = {
    (cirq.XPowGate, -0.5): _builtin_gates.RX,
    (cirq.YPowGate, -0.5): _builtin_gates.RY,
    (cirq.ZPowGate, -0.5): _builtin_gates.RZ,
    (cirq.HPowGate, 0): _builtin_gates.RH,
    (cirq.ZPowGate, 0): _builtin_gates.PHASE,
    (cirq.CZPowGate, 0): _builtin_gates.CPHASE,
    (cirq.XXPowGate, -0.5): _builtin_gates.XX,
    (cirq.YYPowGate, -0.5): _builtin_gates.YY,
    (cirq.ZZPowGate, -0.5): _builtin_gates.ZZ,
    (cirq.ISwapPowGate, 0.0): _builtin_gates.XY,
}

CIRQ_GATE_SPECIAL_CASES = {cirq.CSWAP: _builtin_gates.SWAP.controlled(1)}

qubit_index = attrgetter("x")


@overload
def export_to_cirq(gate: _gates.Gate) -> cirq.Gate:
    pass


@overload
def export_to_cirq(gate_operation: _gates.GateOperation) -> cirq.GateOperation:
    pass


@overload
def export_to_cirq(circuit: _circuit.Circuit) -> cirq.Circuit:
    pass


def export_to_cirq(obj):
    """Export given native Zquantum object to its Cirq equivalent.

    This should be primarily used with Circuit objects, but
    also works for builtin gates and gate operations.

    Exporting of user-defined gates is atm not supported.
    """
    # We need a facade wrapper becase mypy does not yet support overloads for
    # functools.singledispatch. See mypy #8356.

    return _export_to_cirq(obj)


@singledispatch
def _export_to_cirq(obj):
    """Export given native Zquantum object to its Cirq equivalent.

    This should be primarily used with Circuit objects, but
    also works for builtin gates and gate operations.

    Exporting of user-defined gates is atm not supported.
    """
    raise NotImplementedError(f"{obj} can't be exported to Cirq object.")


@_export_to_cirq.register
def _export_matrix_factory_gate_to_cirq(gate: _gates.MatrixFactoryGate) -> cirq.Gate:
    try:
        cirq_factory = ZQUANTUM_BUILTIN_GATE_NAME_TO_CIRQ_GATE[gate.name]
        cirq_params = (
            float(param) if isinstance(param, sympy.Expr) and param.is_Float else param
            for param in gate.params
        )
        return cirq_factory(*cirq_params) if gate.params else cirq_factory
    except KeyError:
        raise NotImplementedError(f"Gate {gate} can't be exported to Cirq.")


@_export_to_cirq.register
def _export_controlled_gate_to_cirq(gate: _gates.ControlledGate) -> cirq.Gate:
    return _export_to_cirq(gate.wrapped_gate).controlled(gate.num_control_qubits)


@_export_to_cirq.register
def _export_dagger_to_cirq(gate: _gates.Dagger) -> cirq.Gate:
    return cirq.inverse(_export_to_cirq(gate.wrapped_gate))


@_export_to_cirq.register
def _export_gate_operation_to_cirq(
    operation: _gates.GateOperation,
) -> cirq.GateOperation:
    return _export_to_cirq(operation.gate)(
        *map(cirq.LineQubit, operation.qubit_indices)
    )


@_export_to_cirq.register
def _export_circuit_to_cirq(circuit: _circuit.Circuit) -> cirq.Circuit:
    return cirq.Circuit(
        [_export_to_cirq(operation) for operation in circuit.operations]
    )


def import_from_cirq(obj):
    """Import given Cirq object, converting it to its ZQuantum counterpart.

    Gates corresponding to ZQuantum built-in gates, operations on such gates and
    circuits composed of such gates will use the native definitions, e.g. `cirq.X` will
    become `circuits.X`.

    Importing gates from Cirq that don't have built-in counterparts in ZQuantum will
    result in custom gates. See `help(zquantum.core.circuits)` for examples of
    custom gates.

    Also note that only objects using only LineQubits are supported, as currently there
    is no notion of GridQubit in ZQuantum circuits.
    """
    return _import_from_cirq(obj)


@dataclass
class NonNativeGate:
    matrix: np.ndarray
    cirq_class: type


def _import_non_built_in_gate(gate) -> NonNativeGate:
    try:
        matrix = cirq.unitary(gate)
    except TypeError as e:
        raise NotImplementedError(
            f"Can't import gate {gate} from cirq, even as a custom definition"
        ) from e

    return NonNativeGate(matrix=matrix, cirq_class=type(gate))


@singledispatch
def _import_from_cirq(obj):
    try:
        return CIRQ_GATE_SPECIAL_CASES[obj]
    except KeyError:
        return _import_non_built_in_gate(obj)


@_import_from_cirq.register
def _convert_qasm_u_gate_to_zquantum_gate(
    ugate: cirq.circuits.qasm_output.QasmUGate,
) -> _gates.Gate:
    angles = (
        exponent_to_angle(angle) for angle in (ugate.theta, ugate.phi, ugate.lmda)
    )
    return _builtin_gates.U3(*angles)


@_import_from_cirq.register
def _convert_eigengate_to_zquantum_gate(
    eigengate: cirq.EigenGate,
) -> Union[_gates.Gate, NonNativeGate]:
    key = (type(eigengate), eigengate.global_shift, eigengate.exponent)
    try:
        return EIGENGATE_SPECIAL_CASES[key]
    except KeyError:
        pass

    try:
        return EIGENGATE_ROTATIONS[key[0:2]](exponent_to_angle(eigengate.exponent))
    except KeyError:
        pass

    return _import_non_built_in_gate(eigengate)


@_import_from_cirq.register
def _convert_cirq_identity_gate_to_zquantum_gate(
    identity_gate: cirq.IdentityGate,
) -> _gates.Gate:
    return _builtin_gates.I


@_import_from_cirq.register
def _import_cirq_controlled_gate(controlled_gate: cirq.ControlledGate) -> _gates.Gate:
    return _import_from_cirq(controlled_gate.sub_gate).controlled(
        controlled_gate.num_controls()
    )


def _hash_hex(bytes_):
    return hashlib.sha256(bytes_).hexdigest()


def _gen_custom_gate_name(gate_cls, matrix: np.ndarray):
    matrix_hash = _hash_hex(matrix.tobytes())
    return f"{gate_cls.__name__}.{matrix_hash}"


@_import_from_cirq.register(cirq.GateOperation)
@_import_from_cirq.register(cirq.ControlledOperation)
def _convert_gate_operation_to_zquantum(operation) -> _gates.GateOperation:
    if not all(isinstance(qubit, cirq.LineQubit) for qubit in operation.qubits):
        raise NotImplementedError(
            f"Failed to import {operation}. Grid qubits are not yet supported."
        )

    imported_gate = _import_from_cirq(operation.gate)
    qubit_indices = map(qubit_index, operation.qubits)

    if isinstance(imported_gate, NonNativeGate):
        custom_gate = _gates.CustomGateDefinition(
            gate_name=_gen_custom_gate_name(
                imported_gate.cirq_class, imported_gate.matrix
            ),
            matrix=sympy.Matrix(imported_gate.matrix),
            params_ordering=(),
        )
        return custom_gate()(*qubit_indices)
    else:
        return imported_gate(*qubit_indices)


@_import_from_cirq.register
def _import_circuit_from_cirq(circuit: cirq.Circuit) -> _circuit.Circuit:
    return _circuit.Circuit(
        [_import_from_cirq(op) for op in chain.from_iterable(circuit.moments)]
    )
