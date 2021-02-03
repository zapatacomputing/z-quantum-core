import pytest
import os
import numpy as np
from numpy.linalg import eigvalsh

from openfermion import QubitOperator, InteractionOperator, get_sparse_operator, get_ground_state

from zquantum.core.openfermion import save_qubit_operator_set, load_qubit_operator_set, save_interaction_operator, load_interaction_operator, load_qubit_operator

from operators import concatenate_qubit_operator_lists, get_one_qubit_hydrogen_hamiltonian

from math import isclose

h2_hamiltonian_grouped = [
    QubitOperator("-0.04475014401986127 [X0 X1 Y2 Y3]"),
    QubitOperator("0.04475014401986127 [X0 Y1 Y2 X3]"),
    QubitOperator("0.04475014401986127 [Y0 X1 X2 Y3]"),
    QubitOperator("-0.04475014401986127 [Y0 Y1 X2 X3]"),
    QubitOperator(
        """0.17771287459806312 [Z0] + 
         0.1705973832722407 [Z0 Z1] + 
         0.12293305054268083 [Z0 Z2] + 
         0.1676831945625421 [Z0 Z3] + 
         0.17771287459806312 [Z1] + 
         0.1676831945625421 [Z1 Z2] + 
         0.12293305054268083 [Z1 Z3] + 
         -0.24274280496459985 [Z2] + 
         0.17627640802761105 [Z2 Z3] + 
         -0.24274280496459985 [Z3]"""
    ),
]


def test_concatenate_qubit_operator_lists():

    group_A = h2_hamiltonian_grouped[:3]
    group_B = h2_hamiltonian_grouped[3:]

    save_qubit_operator_set(group_A, "groupA.json")
    save_qubit_operator_set(group_B, "groupB.json")

    concatenate_qubit_operator_lists("groupA.json", "groupB.json")

    group_read = load_qubit_operator_set("concatenated-qubit-operators.json")

    os.remove("groupA.json")
    os.remove("groupB.json")
    os.remove("concatenated-qubit-operators.json")

    assert group_read == h2_hamiltonian_grouped

def test_get_one_qubit_hydrogen_hamiltonian():

    # Interaction Hamiltonian of H2 (0.74 A, STO-3G) generated using openfermion-psi4
    constant = 0.7151043387432434
    # Load one body tensor from .npz file
    file_handle_1b = np.load('h2_one_body_tensor.npz')
    one_body_tensor = file_handle_1b.get('arr_0')
    # Load one body tensor from .npz file
    file_handle_2b = np.load('h2_two_body_tensor.npz')
    two_body_tensor = file_handle_2b.get('arr_0')
    # Assemble interaction operator
    h2_hamiltonian_int = InteractionOperator(constant=constant, one_body_tensor=one_body_tensor, two_body_tensor=two_body_tensor)
    save_interaction_operator(h2_hamiltonian_int, 'interaction-operator.json')

    get_one_qubit_hydrogen_hamiltonian('interaction-operator.json')
    h2_1qubit = load_qubit_operator('qubit-operator.json')
    h2_1qubit_sparse = get_sparse_operator(h2_1qubit, n_qubits=1)
    h2_1qubit_dense = h2_1qubit_sparse.toarray()
    print(h2_1qubit_dense)

    e_1q = eigvalsh(h2_1qubit_dense)

    gs_4q = get_ground_state(get_sparse_operator(h2_hamiltonian_int))

    assert isclose(e_1q[0], gs_4q[0])

    os.remove('interaction-operator.json')
    os.remove('qubit-operator.json')
