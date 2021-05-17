from typing import Optional, Union, cast

import numpy as np

from ._builtin_gates import GatePrototype
from ._circuit import Circuit
from ._gates import Gate


def create_layer_of_gates(
    number_of_qubits: int,
    gate_factory: Union[GatePrototype, Gate],
    parameters: Optional[np.ndarray] = None,
) -> Circuit:
    """Creates a circuit consisting of a layer of single-qubit gates acting on all
    qubits.

    Args:
        number_of_qubits: number of qubits in the circuit
        gate_name: the single qubit gate to be applied to each qubit
        parameters: parameters of the single-qubit gates

    Returns:
        circuit: Created circuit.
    """
    circuit = Circuit()

    if parameters is not None:
        assert len(parameters) == number_of_qubits
        gate_factory = cast(GatePrototype, gate_factory)
        for i in range(number_of_qubits):
            circuit += gate_factory(parameters[i])(i)
    else:
        gate_factory = cast(Gate, gate_factory)
        for i in range(number_of_qubits):
            circuit += gate_factory(i)

    return circuit
