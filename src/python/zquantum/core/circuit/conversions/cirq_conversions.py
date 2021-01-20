from functools import singledispatch
from ...circuit.gates import (
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    PHASE,
    T,
    I,
    H,
    CZ,
    CNOT,
    CPHASE,
    SWAP,
    Dagger,
)


@singledispatch
def convert_to_cirq(obj):
    raise NotImplementedError(f"Cannot convert {obj} to cirq object.")


@singledispatch
def convert_from_cirq(obj):
    raise NotImplementedError(f"Conversion from cirq to Orquestra not supported for {obj}.")



