from typing import Iterable

import pyquil
import sympy

from .. import _gates
from .. import _builtin_gates


def _n_qubits_by_ops(ops: Iterable[_gates.GateOperation]):
    try:
        return max(op.qubit_indices for op in ops)
    except ValueError:
        return 0


def import_from_pyquil(program: pyquil.Program):
    ops = [_import_gate(instr) for instr in program.instructions if isinstance(instr, pyquil.gates.Gate)]
    return _gates.Circuit(ops, _n_qubits_by_ops(ops))


def _import_gate(gate: pyquil.gates.Gate):
    pass


def export_to_pyquil(circuit: _gates.Circuit) -> pyquil.Program:
    var_declarations = map(_symbol_declaration, circuit.free_symbols)
    gate_instructions = [_export_gate(op.gate, op.qubit_indices) for op in circuit.operations]
    return pyquil.Program([*var_declarations, *gate_instructions])


def _symbol_declaration(symbol: sympy.Symbol):
    return pyquil.quil.Declare(str(symbol))


def _export_gate(gate: _gates.Gate, qubit_indices):
    try:
        return _export_gate_via_mapping(gate, qubit_indices)
    except ValueError:
        pass

    raise NotImplementedError()


def _pyquil_gate_by_name(name):
    return getattr(pyquil.gates, name)


def _export_gate_via_mapping(gate: _gates.Gate, qubit_indices):
    try:
        pyquil_fn = _pyquil_gate_by_name(gate.name)
    except KeyError:
        raise ValueError()

    return pyquil_fn(*qubit_indices)
