from typing import Tuple

import numpy as np
import numpy.random

from . import _builtin_gates
from ._circuit import Circuit

ONE_QUBIT_NO_PARAMS_GATES = [
    _builtin_gates.X,
    _builtin_gates.Y,
    _builtin_gates.Z,
    _builtin_gates.H,
    _builtin_gates.S,
    _builtin_gates.T,
]

ONE_QUBIT_ONE_PARAM_GATES = [
    _builtin_gates.RX,
    _builtin_gates.RY,
    _builtin_gates.RZ,
    _builtin_gates.PHASE,
]

TWO_QUBITS_NO_PARAMS_GATES = [
    _builtin_gates.CNOT,
    _builtin_gates.CZ,
    _builtin_gates.SWAP,
]

TWO_QUBITS_ONE_PARAM_GATES = [
    _builtin_gates.CPHASE,
]


def create_random_circuit(
    n_qubits: int,
    n_gates: int,
    rng: np.random.Generator,
) -> Circuit:
    """Generates random circuit acting on nqubits with ngates for testing purposes.
    The resulting circuit it saved to file in JSON format under 'circuit.json'.

    Args:
        n_qubits: The number of qubits in the circuit
        n_gates: The number of gates in the circuit
        rng: Numpy random generator

    Returns:
        Generated circuit.
    """
    # Initialize all gates in set, not including RH or ZXZ

    all_gates_lists = [
        ONE_QUBIT_NO_PARAMS_GATES,
        TWO_QUBITS_NO_PARAMS_GATES,
        ONE_QUBIT_ONE_PARAM_GATES,
        TWO_QUBITS_ONE_PARAM_GATES,
    ]

    # Loop to add gates to circuit
    circuit = Circuit()
    for gate_i in range(n_gates):
        # Pick gate type
        gates_list = rng.choice(all_gates_lists)
        gate = rng.choice(gates_list)

        # Pick qubit to act on (control if two qubit gate)
        qubits: Tuple[int, ...]
        if gates_list in [ONE_QUBIT_NO_PARAMS_GATES, ONE_QUBIT_ONE_PARAM_GATES]:
            index = rng.choice(range(n_qubits))
            qubits = (int(index),)
        else:
            indices = rng.choice(range(n_qubits), size=2, replace=False)
            qubits = tuple(int(i) for i in indices)

        if gates_list in [ONE_QUBIT_ONE_PARAM_GATES, TWO_QUBITS_ONE_PARAM_GATES]:
            param = rng.uniform(-np.pi, np.pi, size=1)
            gate = gate(float(param))

        circuit += gate(*qubits)
    return circuit
