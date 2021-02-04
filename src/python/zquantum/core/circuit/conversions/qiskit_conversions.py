from functools import singledispatch
from typing import Tuple, List, Union

import qiskit
from zquantum.core.circuit import X, Y, Z, I, T, H, Gate, Circuit, CZ, CNOT, SWAP, ISWAP, RX, RY, RZ, PHASE, CPHASE, XX, \
    YY, ZZ

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
    ZZ: qiskit.extensions.RZZGate
}


QISKIT_TO_ORQUESTRA_MAPPING = {
    value: key for key, value in ORQUESTRA_TO_QISKIT_MAPPING.items()
}


def qiskit_qubit(index: int, num_qubits_in_circuit: int) -> qiskit.circuit.Qubit:
    return qiskit.circuit.Qubit(
        qiskit.circuit.QuantumRegister(num_qubits_in_circuit, "q"), index
    )


def convert_from_qiskit(
    obj: Union[QiskitOperation, qiskit.QuantumCircuit]
) -> Union[Gate, Circuit]:
    if isinstance(obj, tuple):
        return convert_operation_from_qiskit(obj)
    else:
        raise NotImplementedError()


def convert_operation_from_qiskit(operation: QiskitOperation) -> Gate:
    try:
        qiskit_op, qiskit_qubits, _ = operation
        orquestra_gate_cls = QISKIT_TO_ORQUESTRA_MAPPING[type(qiskit_op)]
        return orquestra_gate_cls(
            *(qubit.index for qubit in reversed(qiskit_qubits)),
            *qiskit_op.params
        )
    except KeyError:
        raise NotImplementedError(
            f"Cannot convert {operation} to Orquestra, unknown operation."
        )


@singledispatch
def convert_to_qiskit(obj, num_qubits_in_circuit: int):
    raise NotImplementedError(f"Convertion of {obj} to qiskit is not supported.")


@convert_to_qiskit.register
def convert_orquestra_gate_to_qiskit(
    gate: Gate, num_qubits_in_circuit: int
) -> QiskitOperation:
    try:
        qiskit_qubits = [
            qiskit_qubit(qubit, num_qubits_in_circuit)
            for qubit in reversed(gate.qubits)
        ]
        qiskit_gate_cls = ORQUESTRA_TO_QISKIT_MAPPING[type(gate)]
        return qiskit_gate_cls(*gate.params), qiskit_qubits, []
    except KeyError:
        raise NotImplementedError(f"Conversion of {gate} to Qiskit is not supported.")
