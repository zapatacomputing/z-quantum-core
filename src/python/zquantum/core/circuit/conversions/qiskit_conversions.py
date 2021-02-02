from typing import Tuple, List, Union

import qiskit
from zquantum.core.circuit import X, Y, Z, I, T, H, Gate, Circuit

QiskitOperation = Tuple[
    qiskit.circuit.Instruction, List[qiskit.circuit.Qubit], List[qiskit.circuit.Clbit]
]


def convert_from_qiskit(
    obj: Union[QiskitOperation, qiskit.QuantumCircuit]
) -> Union[Gate, Circuit]:
    pass


def convert_to_qiskit(obj, num_qubits_in_circuit: int):
    pass
