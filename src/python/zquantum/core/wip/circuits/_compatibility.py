from typing import Union

from ...circuit import Circuit as OldCircuit
from ._circuit import Circuit
from .conversions.cirq_conversions import import_from_cirq, export_to_cirq
from .conversions.qiskit_conversions import export_to_qiskit
from .conversions.pyquil_conversions import export_to_pyquil


AnyCircuit = Union[OldCircuit, Circuit]


def new_circuit_from_old_circuit(old: OldCircuit) -> Circuit:
    return import_from_cirq(old.to_cirq())


def old_circuit_from_new_circuit(new: Circuit) -> OldCircuit:
    return OldCircuit(export_to_cirq(new))
    # return OldCircuit(export_to_qiskit(new))
    # return OldCircuit(export_to_qiskit(new))


def ensure_old_circuit(circuit: AnyCircuit) -> OldCircuit:
    if isinstance(circuit, Circuit):
        return old_circuit_from_new_circuit(circuit)
    else:
        return circuit
