"""Circuit utilities."""
from openfermion import QubitOperator
from . import (
    Circuit,
    Gate,
    Qubit,
)
from typing import List, Tuple


def create_circuits_from_qubit_operator(qubit_operator: QubitOperator) -> List[Circuit]:
    """Creates a list of zquantum.core.Circuit objects from the Pauli terms of a QubitOperator

    Args:
        qubit_operator: QubitOperator: qubit operator for which the Pauli terms are converted into Circuits.

    
    Return:
        circuit_set (list): a list of core.Circuit objects

    """
    
    # Get the Pauli terms, ignoring coefficients
    pauli_terms = list(qubit_operator.terms.keys())

    circuit_set = []

    # Loop over Pauli terms and populate circuit set list
    for term in pauli_terms:
        
        circuit = Circuit()
        pauli_gates = []
        qubits = []

        # Loop over Pauli factors in Pauli term and construct Pauli term circuit    
        for pauli in term:   # loop over pauli operators in an n qubit pauli term
            pauli_index = pauli[0]
            pauli_factor = pauli[1]
            pauli_gates.append(Gate(pauli_factor, qubits=[Qubit(pauli_index)]))
            qubits.append(Qubit(pauli[0]))

        circuit.gates = pauli_gates
        circuit.qubits += qubits

        circuit_set += [circuit]

    return circuit_set