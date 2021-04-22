from typing import Union

from ...circuit import Circuit as OldCircuit
from ._circuit import Circuit
from .conversions.cirq_conversions import import_from_cirq


AnyCircuit = Union[OldCircuit, Circuit]


def new_circuit_from_old_circuit(old: OldCircuit) -> Circuit:
    return import_from_cirq(old.to_cirq())
