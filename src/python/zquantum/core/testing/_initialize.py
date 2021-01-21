from pyquil import Program
from pyquil.gates import *
from pyquil.wavefunction import Wavefunction
import math
import random
import numpy as np
from openfermion.ops import QubitOperator, IsingOperator
from ..circuit import Circuit
from ..utils import RNDSEED


def create_random_circuit(nqubits, ngates, seed=None):
    """Generates random circuit acting on nqubits with ngates for testing purposes.
    The resulting circuit it saved to file in JSON format under 'circuit.json'.

    Args:
        nqubits: integer
            The number of qubits in the circuit
        ngates: integer
            The number of gates in the circuit

        *** OPTIONAL ***
        seed: integer
            The see for the random number generator

    Returns:
        None, a Circuit (core.circuit) object is saved under 'circuit.json'
    """
    # Initialize all gates in set, not including RH or ZXZ
    SING_ZERO = [X, Y, Z, H, S, T]
    SING_ONE = [RX, RY, RZ, PHASE]
    TWO_ZERO = [CNOT, CZ, SWAP]
    TWO_ONE = [CPHASE]

    ALL = [SING_ZERO, TWO_ZERO, SING_ONE, TWO_ONE]

    NUM_QUBITS = range(0, nqubits)

    if seed is not None:
        random.seed(seed)

    # Create empty pyquil circuit
    p = Program()

    # Loop to add gates to pyquil circuit
    for i in range(0, ngates):
        # Pick gate type
        TYPE_OF_GATE = random.choice(ALL)
        gate = random.choice(TYPE_OF_GATE)

        # Pick qubit to act on (control if two qubit gate)
        qubit = random.choice(NUM_QUBITS)

        if TYPE_OF_GATE == SING_ZERO:
            p += gate(qubit)
        elif TYPE_OF_GATE == SING_ONE:
            # Choose random parameter between +/- pi
            param = random.uniform(-math.pi, math.pi)
            p += gate(param, qubit)
        elif TYPE_OF_GATE == TWO_ZERO:
            target = random.choice(NUM_QUBITS)
            while target == qubit:
                # Loop to ensure target =/= control
                target = random.choice(NUM_QUBITS)
            p += gate(qubit, target)
        elif TYPE_OF_GATE == TWO_ONE:
            target = random.choice(NUM_QUBITS)
            while target == qubit:
                # Loop to ensure target =/= control
                target = random.choice(NUM_QUBITS)
            # Choose random parameter between +/- pi
            param = random.uniform(-math.pi, math.pi)
            p += gate(param, qubit, target)

    c = Circuit(p)
    return c


def create_random_qubitop(nqubits, nterms, seed=None):
    """Generates random qubit operator acting on nqubits with nterms for testing purposes.
    The resulting qubit operator it saved to file in JSON format under 'qubitop.json'.

    Args:
        nqubits: integer
            The number of qubits in the qubit operator
        nterms: integer
            The number of terms in the qubit operator

        *** OPTIONAL ***
        seed: integer
            The see for the random number generator

    Returns:
        None, a Qubit Operator (openfermion.QubitOperator) object is saved under 'qubitop.json'
    """
    NUM_QUBITS = range(0, nqubits)

    if seed is not None:
        random.seed(seed)

    # Initialize empty qubit operator
    qubitop = QubitOperator()

    # Loop over number of separate terms in qubit operator
    for i in range(0, nterms):
        # Choose number of paulis to measure in term
        num_paulis = random.choice(range(nqubits + 1))

        # Create empty list of qubits
        qubits = []

        # Create empty term
        full_term = ""

        # Loop over paulis
        for j in range(0, num_paulis):
            # Choose random qubit
            qubit_index = random.choice(NUM_QUBITS)
            while qubit_index in qubits:
                # Ensure qubit not already being measured in this term
                qubit_index = random.choice(NUM_QUBITS)
            qubits.append(qubit_index)

            # Choose pauli
            pauli_gate = random.choice(["X", "Y", "Z"])
            # Construct string
            full_term += pauli_gate + str(qubit_index) + " "
        # Add full term to qubit operator
        qubitop += QubitOperator(full_term)

    return qubitop


def create_random_isingop(nqubits, nterms, seed=None):
    """Generates random ising operator acting on nqubits with nterms for testing purposes.

    Args:
        nqubits: integer
            The number of qubits in the qubit operator
        nterms: integer
            The number of terms in the qubit operator

        *** OPTIONAL ***
        seed: integer
            The see for the random number generator

    Returns:
        an Ising Operator (openfermion.IsingOperator) object
    """
    NUM_QUBITS = range(0, nqubits)

    if seed is not None:
        random.seed(seed)

    # Initialize empty qubit operator
    isingop = IsingOperator()

    # Loop over number of separate terms in qubit operator
    for i in range(0, nterms):
        # Choose number of paulis to measure in term
        num_paulis = random.choice(range(nqubits + 1))

        # Create empty list of qubits
        qubits = []

        # Create empty term
        full_term = ""

        # Loop over paulis
        for j in range(0, num_paulis):
            # Choose random qubit
            qubit_index = random.choice(NUM_QUBITS)
            while qubit_index in qubits:
                # Ensure qubit not already being measured in this term
                qubit_index = random.choice(NUM_QUBITS)
            qubits.append(qubit_index)

            # Choose pauli
            pauli_gate = "Z"
            # Construct string
            full_term += pauli_gate + str(qubit_index) + " "
        # Add full term to qubit operator
        isingop += IsingOperator(full_term)

    return isingop


def create_random_wavefunction(n_qubits, seed=None):
    if seed:
        np.random.seed(seed)

    random_vector = [
        complex(a, b)
        for a, b in zip(np.random.rand(2 ** n_qubits), np.random.rand(2 ** n_qubits))
    ]
    normalization_factor = np.sqrt(np.sum(np.abs(random_vector) ** 2))
    random_vector /= normalization_factor

    return Wavefunction(random_vector)
