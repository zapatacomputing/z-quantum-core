from ..circuits import Circuit as OldCircuit

from . import Circuit
from .conversions.cirq_conversions import import_from_cirq


def new_circuit_from_old_circuit(old: OldCircuit) -> Circuit:
    return import_from_cirq(old.to_cirq())
