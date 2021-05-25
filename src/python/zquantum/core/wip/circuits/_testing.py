import numpy as np
import numpy.random

from . import _builtin_gates
from ._circuit import Circuit


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
    singular_zero = [
        _builtin_gates.X,
        _builtin_gates.Y,
        _builtin_gates.Z,
        _builtin_gates.H,
        _builtin_gates.S,
        _builtin_gates.T,
    ]
    singular_one = [
        _builtin_gates.RX,
        _builtin_gates.RY,
        _builtin_gates.RZ,
        _builtin_gates.PHASE,
    ]
    two_zero = [
        _builtin_gates.CNOT,
        _builtin_gates.CZ,
        _builtin_gates.SWAP,
    ]
    two_one = [
        _builtin_gates.CPHASE,
    ]

    all_gate_factories = [singular_zero, two_zero, singular_one, two_one]

    # Loop to add gates to circuit
    circuit = Circuit()
    for gate_i in range(n_gates):
        # Pick gate type
        gates_list = rng.choice(all_gate_factories)
        gate = rng.choice(gates_list)

        # Pick qubit to act on (control if two qubit gate)

        if gates_list in {singular_zero, singular_one}:
            qubits = (rng.choice(range(n_qubits), size=1),)
        else:
            qubits = rng.choice(range(n_qubits), size=2, replace=False)

        if gates_list in {singular_one, two_one}:
            gate = gate(rng.uniform(-np.pi, np.pi, size=1))

        circuit += gate(*qubits)
    return circuit
