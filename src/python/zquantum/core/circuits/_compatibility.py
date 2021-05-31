import warnings

from ._circuit import Circuit

AnyCircuit = Circuit


def new_circuit_from_old_circuit(old) -> Circuit:
    warnings.warn(
        "Old circuits have been removed, usage of this function should be removed.",
        DeprecationWarning,
    )
    return old
