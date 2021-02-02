from typing import Tuple, List, Union

import qiskit
from zquantum.core.circuit import X, Y, Z, I, T, H, Gate, Circuit

QiskitOperation = Tuple[
    qiskit.circuit.Instruction, List[qiskit.circuit.Qubit], List[qiskit.circuit.Clbit]
]


ORQUESTRA_TO_QISKIT_MAPPING = {
    X: qiskit.extensions.XGate,
    Y: qiskit.extensions.YGate,
    Z: qiskit.extensions.ZGate,
    T: qiskit.extensions.TGate,
    H: qiskit.extensions.HGate,
    I: qiskit.extensions.IGate
}


QISKIT_TO_ORQUESTRA_MAPPING = {
    value: key for key, value in ORQUESTRA_TO_QISKIT_MAPPING.items()
}


def convert_from_qiskit(
    obj: Union[QiskitOperation, qiskit.QuantumCircuit]
) -> Union[Gate, Circuit]:
    if isinstance(obj, tuple):
        return convert_operation_from_qiskit(obj)
    else:
        raise NotImplementedError()


def convert_operation_from_qiskit(instruction: QiskitOperation) -> Gate:
    try:
        qiskit_op, qiskit_qubits, _ = instruction
        orquestra_gate_cls = QISKIT_TO_ORQUESTRA_MAPPING[type(qiskit_op)]
        return orquestra_gate_cls(qiskit_qubits[0].index)
    except KeyError:
        raise NotImplementedError(
            f"Cannot convert {instruction} to Orquestra, unknown operation."
        )


def convert_to_qiskit(obj, num_qubits_in_circuit: int):
    pass
