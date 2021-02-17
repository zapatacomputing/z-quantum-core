"""Conversions between Qiskit and Orquestra objects."""
from functools import singledispatch
from typing import Tuple, List, Union

import qiskit

from .symbolic.qiskit_expressions import expression_from_qiskit, QISKIT_DIALECT
from .symbolic.sympy_expressions import expression_from_sympy, SYMPY_DIALECT
from .symbolic.translations import translate_expression
from .. import (
    X,
    Y,
    Z,
    I,
    T,
    H,
    Gate,
    Circuit,
    CZ,
    CNOT,
    SWAP,
    ISWAP,
    RX,
    RY,
    RZ,
    PHASE,
    CPHASE,
    XX,
    YY,
    ZZ,
    ControlledGate,
)

QiskitOperation = Tuple[
    qiskit.circuit.Instruction, List[qiskit.circuit.Qubit], List[qiskit.circuit.Clbit]
]


ORQUESTRA_TO_QISKIT_MAPPING = {
    X: qiskit.extensions.XGate,
    Y: qiskit.extensions.YGate,
    Z: qiskit.extensions.ZGate,
    T: qiskit.extensions.TGate,
    H: qiskit.extensions.HGate,
    I: qiskit.extensions.IGate,
    CNOT: qiskit.extensions.CXGate,
    CZ: qiskit.extensions.CZGate,
    SWAP: qiskit.extensions.SwapGate,
    ISWAP: qiskit.extensions.iSwapGate,
    RX: qiskit.extensions.RXGate,
    RY: qiskit.extensions.RYGate,
    RZ: qiskit.extensions.RZGate,
    PHASE: qiskit.extensions.PhaseGate,
    CPHASE: qiskit.extensions.CPhaseGate,
    XX: qiskit.extensions.RXXGate,
    YY: qiskit.extensions.RYYGate,
    ZZ: qiskit.extensions.RZZGate,
    # TODO: add CustomGate mapping after we decide on "Gate|GateOperation" split
}


def _make_controlled_gate_factory(gate_cls):
    def _factory(control, *gate_args):
        return ControlledGate(gate_cls(*gate_args), control)

    return _factory


# NOTE:
# Qiskit implements a special subclasses for some of the controlled gate variants:
# https://github.com/Qiskit/qiskit-terra/blob/0.16.4/qiskit/circuit/add_control.py#L72
# We'll support most of them eventually. The development plan is:
# 1. Add mapping for some most common cases (done).
# 2. Support most of the Qiskit controlled gates via our CustomGate when it's reworked.
# 3. Add specialized gate subclasses on our side if there's a need for handling special
#     cases explicitly.


QISKIT_TO_ORQUESTRA_MAPPING = {
    **{
        value: key for key, value in ORQUESTRA_TO_QISKIT_MAPPING.items()
    },
    qiskit.extensions.CRXGate: _make_controlled_gate_factory(RX),
    qiskit.extensions.CRYGate: _make_controlled_gate_factory(RY),
    qiskit.extensions.CRZGate: _make_controlled_gate_factory(RZ),
    qiskit.extensions.CSwapGate: _make_controlled_gate_factory(SWAP),
}


def qiskit_qubit(index: int, num_qubits_in_circuit: int) -> qiskit.circuit.Qubit:
    return qiskit.circuit.Qubit(
        qiskit.circuit.QuantumRegister(num_qubits_in_circuit, "q"), index
    )


def convert_from_qiskit(
    obj: Union[QiskitOperation, qiskit.QuantumCircuit]
) -> Union[Gate, Circuit]:
    """Convert Qiskit object to a corresponding one in Orquestra.

    Args:
        obj: Qiskit object to convert. Currently, only gate operations are supported.

    Returns:
        Orquestra object converted from the `obj`.
    """
    if isinstance(obj, tuple):
        return _convert_operation_from_qiskit(obj)
    elif isinstance(obj, qiskit.QuantumCircuit):
        return _convert_circuit_from_qiskit(obj)
    else:
        raise NotImplementedError(
            f"Cannot convert {obj} to Orquestra"
        )


def _convert_operation_from_qiskit(operation: QiskitOperation) -> Gate:
    try:
        qiskit_op, qiskit_qubits, _ = operation
        orquestra_gate_cls = QISKIT_TO_ORQUESTRA_MAPPING[type(qiskit_op)]
        orquestra_params = [
            translate_expression(intermediate_expr, SYMPY_DIALECT)
            for intermediate_expr in map(expression_from_qiskit, qiskit_op.params)
        ]
        return orquestra_gate_cls(
            *(qubit.index for qubit in qiskit_qubits), *orquestra_params
        )
    except KeyError:
        raise NotImplementedError(
            f"Cannot convert {operation} to Orquestra, unknown operation."
        )


def _convert_circuit_from_qiskit(circuit: qiskit.QuantumCircuit) -> Circuit:
    orq_gates = [convert_from_qiskit(triplet) for triplet in circuit.data]
    return Circuit(gates=orq_gates, n_qubits=circuit.num_qubits)


@singledispatch
def convert_to_qiskit(obj, num_qubits_in_circuit: int) -> QiskitOperation:
    """Convert Orquestra object to a corresponding one in Qiskit.

    Args:
        obj: Object to convert. Currenly, only gates are supported.
        num_qubits_in_circuit: Number of qubits in target Qiskit circuit. It's
            needed because Orquestra gates don't contain information about
            number of qubits in circuit, but Qiskit operations do.

    Returns:
        Qiskit operation converted from the `obj`.
    """
    raise NotImplementedError(f"Convertion of {obj} to qiskit is not supported.")



@convert_to_qiskit.register
def _convert_orquestra_gate_to_qiskit(
    gate: Gate, num_qubits_in_circuit: int
) -> QiskitOperation:
    try:
        qiskit_qubits = [
            qiskit_qubit(qubit, num_qubits_in_circuit)
            for qubit in gate.qubits
        ]
        qiskit_gate_cls = ORQUESTRA_TO_QISKIT_MAPPING[type(gate)]
        qiskit_params = [
            translate_expression(intermediate_expr, QISKIT_DIALECT)
            for intermediate_expr in map(expression_from_sympy, gate.params)
        ]
        return qiskit_gate_cls(*qiskit_params), qiskit_qubits, []
    except KeyError:
        raise NotImplementedError(f"Conversion of {gate} to Qiskit is not supported.")


@convert_to_qiskit.register
def _convert_controlled_gate_to_qiskit(
    gate: ControlledGate, num_qubits_in_circuit: int
) -> QiskitOperation:
    target_gate, _, _ = convert_to_qiskit(gate.target_gate, num_qubits_in_circuit)
    # NOTE: We're assuming that Orquestra only supports singly-controlled gates.
    controlled_gate = target_gate.control(1)
    qiskit_qubits = [qiskit_qubit(qubit, num_qubits_in_circuit) for qubit in gate.qubits]
    return controlled_gate, qiskit_qubits, []


@convert_to_qiskit.register
def _convert_orquestra_circuit_to_qiskit(orq_circuit: Circuit):
    qiskit_circuit = qiskit.QuantumCircuit(orq_circuit.n_qubits)
    qiskit_triplets = [
        convert_to_qiskit(orq_gate, orq_circuit.n_qubits)
        for orq_gate in orq_circuit.gates
    ]
    for qiskit_gate, qiskit_qubits, qiskit_clbits in qiskit_triplets:
        qiskit_circuit.append(qiskit_gate, qiskit_qubits, qiskit_clbits)

    return qiskit_circuit
