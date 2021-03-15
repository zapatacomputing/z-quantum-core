from functools import singledispatch
from typing import Iterable

import pyquil
import sympy

from .. import _gates
from .. import _builtin_gates


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
    zq_gate = _builtin_gates.builtin_gate_by_name(gate.name)
    for modifier in gate.modifiers:
        if modifier == "DAGGER":
            zq_gate = zq_gate.dagger
        elif modifier == "CONTROLLED":
            zq_gate = zq_gate.controlled(1)
    all_qubits = _import_pyquil_qubits(gate.qubits)
    return zq_gate(*all_qubits) if not gate.params else zq_gate(*gate.params)(*all_qubits)


def export_to_pyquil(circuit: _gates.Circuit) -> pyquil.Program:
    var_declarations = map(_symbol_declaration, circuit.free_symbols)
    gate_instructions = [_export_gate(op.gate, op.qubit_indices) for op in circuit.operations]
    return pyquil.Program([*var_declarations, *gate_instructions])


def _symbol_declaration(symbol: sympy.Symbol):
    return pyquil.quil.Declare(str(symbol))


@singledispatch
def _export_gate(gate: _gates.Gate, qubit_indices):
    try:
        return _export_gate_via_name(gate, qubit_indices)
    except ValueError:
        pass

    raise NotImplementedError()


@_export_gate.register
def _export_controlled_gate(gate: _gates.ControlledGate, qubit_indices):
    wrapped_qubit_indices = qubit_indices[gate.num_control_qubits:]
    control_qubit_indices = qubit_indices[0:gate.num_control_qubits]
    exported = _export_gate(gate.wrapped_gate, wrapped_qubit_indices)
    for index in reversed(control_qubit_indices):
        exported = exported.controlled(index)
    return exported


def _pyquil_gate_by_name(name):
    return getattr(pyquil.gates, name)


def _export_gate_via_name(gate: _gates.Gate, qubit_indices):
    try:
        pyquil_fn = _pyquil_gate_by_name(gate.name)
    except KeyError:
        raise ValueError()

    return pyquil_fn(*gate.params, *qubit_indices)
