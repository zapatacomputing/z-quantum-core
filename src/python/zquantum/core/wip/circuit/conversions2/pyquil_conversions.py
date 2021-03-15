from functools import singledispatch, reduce
from typing import Iterable, Mapping

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


def _import_expression(pyquil_expr):
    return translate_expression(expression_from_pyquil(pyquil_expr), SYMPY_DIALECT)


def _export_expression(expr: sympy.Expr):
    return translate_expression(expression_from_sympy(expr), QUIL_DIALECT)


def _import_matrix(pyquil_matrix):
    raise NotImplementedError()


def _export_matrix(matrix: sympy.Matrix):
    return [
        [_export_expression(element) for element in row] for row in matrix.tolist()
    ]


def _import_gate_def(gate_def: pyquil.quilbase.DefGate):
    if gate_def.parameters:
        raise NotImplementedError(f"Can't import parametric custom gate def {gate_def}")

    # TODO: make sure PyQuil checks for gate name uniqueness

    pyquil_params = tuple(map(_import_expression, gate_def.parameters))

    return _gates.CustomGateDefinition(
        gate_name=gate_def.name,
        matrix=_import_matrix(gate_def.matrix),
        params_ordering=pyquil_params,
    )


def import_from_pyquil(program: pyquil.Program):
    custom_defs = {
        gate_def.name: _import_gate_def(gate_def) for gate_def in program.defined_gates
    }
    ops = [
        _import_gate(instr, custom_defs)
        for instr in program.instructions
        if isinstance(instr, pyquil.gates.Gate)
    ]
    return _gates.Circuit(ops, _n_qubits_by_ops(ops), custom_defs.values())


def _import_gate(
    instruction: pyquil.gates.Gate,
    custom_gate_defs: Mapping[str, _gates.CustomGateDefinition],
) -> _gates.GateOperation:
    try:
        return _import_gate_via_name(instruction)
    except ValueError:
        pass

    try:
        return _import_custom_gate(instruction, custom_gate_defs)
    except ValueError:
        pass

    raise NotImplementedError()


def _import_gate_via_name(gate: pyquil.gates.Gate) -> _gates.GateOperation:
    zq_gate_ref = _builtin_gates.builtin_gate_by_name(gate.name)
    if not zq_gate_ref:
        raise ValueError()

    zq_params = tuple(map(_import_expression, gate.params))
    zq_gate = zq_gate_ref(*zq_params) if zq_params else zq_gate_ref

    for modifier in gate.modifiers:
        if modifier == "DAGGER":
            zq_gate = zq_gate.dagger
        elif modifier == "CONTROLLED":
            zq_gate = zq_gate.controlled(1)
    all_qubits = _import_pyquil_qubits(gate.qubits)
    return zq_gate(*all_qubits)


def _import_custom_gate(instruction, custom_gate_defs):
    try:
        gate_def = custom_gate_defs[instruction.name]
    except KeyError:
        raise ValueError()

    zq_params = tuple(map(_import_expression, instruction.params))
    zq_qubits = _import_pyquil_qubits(instruction.qubits)
    return gate_def(*zq_params)(*zq_qubits)


def _import_pyquil_qubits(qubits: Iterable[pyquil.quil.Qubit]):
    return tuple(qubit.index for qubit in qubits)




def _assign_custom_defs(program: pyquil.Program, custom_gate_defs):
    def _reducer(prog: pyquil.Program, gate_def: _gates.CustomGateDefinition):
        pyquil_params = list(map(_export_expression, gate_def.params_ordering))
        return prog.defgate(gate_def.gate_name, _export_matrix(gate_def.matrix), pyquil_params)

    return reduce(_reducer, custom_gate_defs, program)


def export_to_pyquil(circuit: _gates.Circuit) -> pyquil.Program:
    var_declarations = map(_param_declaration, sorted(map(str, circuit.free_symbols)))
    custom_gate_names = {
        gate_def.gate_name for gate_def in circuit.custom_gate_definitions
    }
    gate_instructions = [
        _export_gate(op.gate, op.qubit_indices, custom_gate_names)
        for op in circuit.operations
    ]
    program = pyquil.Program(*[*var_declarations, *gate_instructions])
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
    pyquil_params = list(map(_export_expression, gate.params))
    return (gate.name, pyquil_params) + qubit_indices


@_export_gate.register
def _export_controlled_gate(
    gate: _gates.ControlledGate, qubit_indices, custom_gate_names
):
    wrapped_qubit_indices = qubit_indices[gate.num_control_qubits :]
    control_qubit_indices = qubit_indices[0 : gate.num_control_qubits]
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

    pyquil_params = map(_export_expression, gate.params)
    return pyquil_fn(*pyquil_params, *qubit_indices)
