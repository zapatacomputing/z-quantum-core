from functools import singledispatch
from typing import Iterable

import pyquil
import sympy
import numpy as np

from .. import _gates
from .. import _builtin_gates
from ..symbolic.translations import translate_expression
from ..symbolic.pyquil_expressions import QUIL_DIALECT, expression_from_pyquil
from ..symbolic.sympy_expressions import SYMPY_DIALECT, expression_from_sympy


def _n_qubits_by_ops(ops: Iterable[_gates.GateOperation]):
    try:
        return max(qubit_i for op in ops for qubit_i in op.qubit_indices) + 1
    except ValueError:
        return 0


def import_from_pyquil(program: pyquil.Program):
    ops = [_import_gate(instr) for instr in program.instructions if isinstance(instr, pyquil.gates.Gate)]
    return _gates.Circuit(ops, _n_qubits_by_ops(ops))


def _import_gate(gate: pyquil.gates.Gate) -> _gates.GateOperation:
    try:
        return _import_gate_via_name(gate)
    except ValueError:
        pass

    raise NotImplementedError()


def _import_pyquil_qubits(qubits: Iterable[pyquil.quil.Qubit]):
    return tuple(qubit.index for qubit in qubits)


def _import_gate_via_name(gate: pyquil.gates.Gate) -> _gates.GateOperation:
    zq_params = tuple(
        translate_expression(expression_from_pyquil(param), SYMPY_DIALECT)
        for param in gate.params
    )
    zq_gate = (
        _builtin_gates.builtin_gate_by_name(gate.name) if not zq_params
        else _builtin_gates.builtin_gate_by_name(gate.name)(*zq_params)
    )
    for modifier in gate.modifiers:
        if modifier == "DAGGER":
            zq_gate = zq_gate.dagger
        elif modifier == "CONTROLLED":
            zq_gate = zq_gate.controlled(1)
    all_qubits = _import_pyquil_qubits(gate.qubits)
    return zq_gate(*all_qubits)


def _sympy_matrix_to_numpy(matrix: sympy.Matrix):
    return np.array(matrix, dtype=complex)


def _assign_custom_defs(program: pyquil.Program, custom_gate_defs):
    p = program
    for gate_def in custom_gate_defs:
        p = p.defgate(gate_def.gate_name, _sympy_matrix_to_numpy(gate_def.matrix))
    return p


def export_to_pyquil(circuit: _gates.Circuit) -> pyquil.Program:
    var_declarations = map(_param_declaration, sorted(map(str, circuit.free_symbols)))
    custom_gate_names = {gate_def.gate_name for gate_def in circuit.custom_gate_definitions}
    gate_instructions = [_export_gate(op.gate, op.qubit_indices, custom_gate_names) for op in circuit.operations]
    program = pyquil.Program([*var_declarations, *gate_instructions])
    return _assign_custom_defs(program, circuit.custom_gate_definitions)


def _param_declaration(param_name: str):
    return pyquil.quil.Declare(param_name, "REAL")


@singledispatch
def _export_gate(gate: _gates.Gate, qubit_indices, custom_gate_names):
    try:
        return _export_gate_via_name(gate, qubit_indices, custom_gate_names)
    except ValueError:
        pass

    try:
        return _export_custom_gate(gate, qubit_indices, custom_gate_names)
    except ValueError:
        pass

    raise NotImplementedError()


def _export_custom_gate(gate: _gates.Gate, qubit_indices, custom_gate_names):
    if gate.name not in custom_gate_names:
        raise ValueError()
    return (gate.name,) + qubit_indices


@_export_gate.register
def _export_controlled_gate(gate: _gates.ControlledGate, qubit_indices, custom_gate_names):
    wrapped_qubit_indices = qubit_indices[gate.num_control_qubits:]
    control_qubit_indices = qubit_indices[0:gate.num_control_qubits]
    exported = _export_gate(gate.wrapped_gate, wrapped_qubit_indices, custom_gate_names)
    for index in reversed(control_qubit_indices):
        exported = exported.controlled(index)
    return exported


@_export_gate.register
def _export_dagger(gate: _gates.Dagger, qubit_indices, custom_gate_names):
    return _export_gate(gate.wrapped_gate, qubit_indices, custom_gate_names).dagger()


def _pyquil_gate_by_name(name):
    return getattr(pyquil.gates, name)


def _export_gate_via_name(gate: _gates.Gate, qubit_indices, custom_gate_names):
    try:
        pyquil_fn = _pyquil_gate_by_name(gate.name)
    except AttributeError:
        raise ValueError()

    pyquil_params = [
        translate_expression(expression_from_sympy(param), QUIL_DIALECT)
        for param in gate.params
    ]
    return pyquil_fn(*pyquil_params, *qubit_indices)
