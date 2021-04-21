from ...circuit import Circuit as OldCircuit

from ._circuit import Circuit as NewCircuit
from .conversions.cirq_conversions import import_from_cirq


def new_circuit_from_old_circuit(old: OldCircuit) -> NewCircuit:
    return import_from_cirq(old.to_cirq())
