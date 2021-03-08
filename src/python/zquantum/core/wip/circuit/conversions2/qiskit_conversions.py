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


def _zquantum_expr_from_qiskit(expr):
    intermediate = expression_from_qiskit(expr)
    return translate_expression(intermediate, SYMPY_DIALECT)


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

QISKIT_ZQUANTUM_GATE_MAP = {
    **{
        cls: name for name, cls in ZQUANTUM_QISKIT_GATE_MAP.items()
    },
    # qiskit.circuit.library.CRXGate:
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
def _convert_controlled_gate_to_qiskit(
    gate: g.ControlledGate, applied_qubit_indices, n_qubits_in_circuit
):
    target_indices = applied_qubit_indices[gate.num_control_qubits:]
    target_gate, _, _ = _convert_gate_to_qiskit(
        gate.wrapped_gate, target_indices, n_qubits_in_circuit
    )
    controlled_gate = target_gate.control(gate.num_control_qubits)
    qiskit_qubits = [
        qiskit_qubit(qubit_i, n_qubits_in_circuit) for qubit_i in applied_qubit_indices
    ]
    return controlled_gate, qiskit_qubits, []


def _make_gate_instance(gate_ref, gate_params) -> g.Gate:
    """Returns a gate instance that's applicable to qubits.
    For non-parametric gate refs like X, returns just the `X`
    For parametric gate factories like `RX`, returns the produced gate, like `RX(0.2)`
    """
    if g.is_non_parametric(gate_ref):
        return gate_ref
    else:
        return gate_ref(*gate_params)


def _convert_qiskit_triplet_to_op(qiskit_triplet: QiskitOperation) -> g.GateOperation:
    qiskit_op, qiskit_qubits, _ = qiskit_triplet
    try:
        zquantum_name = QISKIT_ZQUANTUM_GATE_MAP[type(qiskit_op)]
        gate_ref = g.builtin_gate_by_name(zquantum_name)
    except KeyError:
        raise NotImplementedError(f"Conversion of {qiskit_op} from Qiskit is unsupported.")

    # values to consider:
    # - gate matrix parameters (only parametric gates)
    # - gate application indices (all gates)
    zquantum_params = [_zquantum_expr_from_qiskit(param) for param in qiskit_op.params]
    qubit_indices = [qubit.index for qubit in qiskit_qubits]
    gate = _make_gate_instance(gate_ref, zquantum_params)
    return g.GateOperation(
        gate=gate,
        qubit_indices=tuple(qubit_indices)
    )


def convert_to_qiskit(circuit: g.Circuit) -> qiskit.QuantumCircuit:
    q_circuit = qiskit.QuantumCircuit(circuit.n_qubits)
    q_triplets = [
        _convert_gate_to_qiskit(gate_op.gate, gate_op.qubit_indices, circuit.n_qubits)
        for gate_op in circuit.operations
    ]
    for q_gate, q_qubits, q_clbits in q_triplets:
        q_circuit.append(q_gate, q_qubits, q_clbits)
    return q_circuit


def convert_from_qiskit(circuit: qiskit.QuantumCircuit) -> g.Circuit:
    q_ops = [_convert_qiskit_triplet_to_op(triplet) for triplet in circuit.data]
    return g.Circuit(operations=q_ops, n_qubits=circuit.num_qubits)
