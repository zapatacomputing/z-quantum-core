from typing import Iterable

import pyquil
import sympy

from .. import _gates


def _n_qubits_by_ops(ops: Iterable[_gates.GateOperation]):
    try:
        return max(op.qubit_indices for op in ops)
    except ValueError:
        return 0


def import_from_pyquil(circuit: pyquil.Program):
    ops = []
    return _gates.Circuit(ops, _n_qubits_by_ops(ops))


def export_to_pyquil(circuit: _gates.Circuit) -> pyquil.Program:
    var_declarations = map(_symbol_declaration, circuit.free_symbols)
    gate_instructions = [_export_gate(op.gate, op.qubit_indices) for op in circuit.operations]
    return pyquil.Program([*var_declarations, *gate_instructions])


def _symbol_declaration(symbol: sympy.Symbol):
    return pyquil.quil.Declare(str(symbol))


def _export_gate(gate: _gates.Gate, qubit_indices):
    raise NotImplementedError()
