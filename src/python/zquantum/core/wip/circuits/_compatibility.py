import zquantum.core.circuit as old_circuit

from . import Circuit, import_from_cirq


def new_circuit_from_old_circuit(old: old_circuit.Circuit) -> Circuit:
    return import_from_cirq(old.to_cirq())
