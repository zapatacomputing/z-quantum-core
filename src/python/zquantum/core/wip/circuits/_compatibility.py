from typing import Union

from ...circuit import Circuit as OldCircuit
from ._circuit import Circuit as NewCircuit
from .conversions.cirq_conversions import import_from_cirq

AnyCircuit = Union[OldCircuit, NewCircuit]


def new_circuit_from_old_circuit(old: OldCircuit) -> NewCircuit:
    if not old.qubits:
        if old.gates:
            raise ValueError(
                "Circuit has defined gates but does not have defined qubits."
            )
        else:
            return NewCircuit([])

    pre_new_circuit = import_from_cirq(old.to_cirq())
    n_qubits = max([qubit.index for qubit in old.qubits]) + 1
    return NewCircuit(pre_new_circuit.operations, n_qubits=n_qubits)
