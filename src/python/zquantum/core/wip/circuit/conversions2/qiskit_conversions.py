from typing import Tuple, List
from functools import singledispatch

import qiskit

from .. import _gates as g
from ..symbolic.sympy_expressions import expression_from_sympy, SYMPY_DIALECT
from ..symbolic.qiskit_expressions import expression_from_qiskit, QISKIT_DIALECT
from ..symbolic.translations import translate_expression

QiskitOperation = Tuple[
    qiskit.circuit.Instruction, List[qiskit.circuit.Qubit], List[qiskit.circuit.Clbit]
]


def qiskit_qubit(index: int, num_qubits_in_circuit: int) -> qiskit.circuit.Qubit:
    return qiskit.circuit.Qubit(
        qiskit.circuit.QuantumRegister(num_qubits_in_circuit, "q"), index
    )


def _qiskit_expr_from_zquantum(expr):
    intermediate = expression_from_sympy(expr)
    return translate_expression(intermediate, QISKIT_DIALECT)


ZQUANTUM_QISKIT_GATE_MAP = {
    "X": qiskit.circuit.library.XGate,
    "Y": qiskit.circuit.library.YGate,
    "Z": qiskit.circuit.library.ZGate,
    "T": qiskit.circuit.library.TGate,
    "H": qiskit.circuit.library.HGate,
    "I": qiskit.circuit.library.IGate,
    "CNOT": qiskit.circuit.library.CXGate,
    "CZ": qiskit.circuit.library.CZGate,
    "SWAP": qiskit.circuit.library.SwapGate,
    "ISWAP": qiskit.circuit.library.iSwapGate,
    "RX": qiskit.circuit.library.RXGate,
    "RY": qiskit.circuit.library.RYGate,
    "RZ": qiskit.circuit.library.RZGate,
    "PHASE": qiskit.circuit.library.PhaseGate,
    "CPHASE": qiskit.circuit.library.CPhaseGate,
    "XX": qiskit.circuit.library.RXXGate,
    "YY": qiskit.circuit.library.RYYGate,
    "ZZ": qiskit.circuit.library.RZZGate,
}


@singledispatch
def _convert_gate_to_qiskit(gate, applied_qubit_indices, n_qubits_in_circuit):
    qiskit_params = [_qiskit_expr_from_zquantum(param) for param in gate.params]
    qiskit_qubits = [
        qiskit_qubit(qubit_i, n_qubits_in_circuit) for qubit_i in applied_qubit_indices
    ]
    try:
        qiskit_cls = ZQUANTUM_QISKIT_GATE_MAP[gate.name]
        return qiskit_cls(*qiskit_params), qiskit_qubits, []
    except KeyError:
        raise NotImplementedError(f"Conversion of {gate} to Qiskit is unsupported.")


@_convert_gate_to_qiskit.register
def _convert_controlled_gate_to_qiskit(gate: g.ControlledGate, applied_qubit_indices, n_qubits_in_circuit):
    target_indices = applied_qubit_indices[gate.num_control_qubits:]
    target_gate, _, _ = _convert_gate_to_qiskit(gate.wrapped_gate, target_indices, n_qubits_in_circuit)
    controlled_gate = target_gate.control(gate.num_control_qubits)
    qiskit_qubits = [qiskit_qubit(qubit_i, n_qubits_in_circuit) for qubit_i in applied_qubit_indices]
    return controlled_gate, qiskit_qubits, []


def convert_to_qiskit(circuit: g.Circuit) -> qiskit.QuantumCircuit:
    if circuit.free_symbols:
        raise NotImplementedError(
            "Converting parametrized circuits to Qiskit is unsupported"
        )

    q_circuit = qiskit.QuantumCircuit(circuit.n_qubits)
    q_triplets = [
        _convert_gate_to_qiskit(gate_op.gate, gate_op.qubit_indices, circuit.n_qubits)
        for gate_op in circuit.operations
    ]
    for q_gate, q_qubits, q_clbits in q_triplets:
        q_circuit.append(q_gate, q_qubits, q_clbits)
    return q_circuit
