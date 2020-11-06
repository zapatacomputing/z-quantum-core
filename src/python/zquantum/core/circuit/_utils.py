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


def create_list_of_full_weight_paulis(number_of_qubits):

    """Creates a list of Pauli strings for a specified qubit number

    Args:
        number_of_qubits: number of qubits

    
    Return:
        pauli_strings (list of strings): a list of Pauli strings

    """

    # Initialize Pauli letter list and string list
    pauli_list = ["", "X", "Y", "Z"]
    current_list = ["["]
    
    # Recursively build elements in list
    for qubit_index in range(number_of_qubits):
        new_list = []
        for j, term in enumerate(current_list):
            for pauli in pauli_list:
                # Append letter and index if non-trivial
                if pauli:
                    new_list.append(term+pauli+f"{qubit_index}"+" ")
                else:
                    new_list.append(term+" ")

            current_list = new_list

    # Replace final space with bracket
    final_list = []
    for j, term in enumerate(current_list):
        final_list.append(term[:-1]+"]")
    
    return final_list