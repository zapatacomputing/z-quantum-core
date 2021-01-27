import pytest
import os

from openfermion import QubitOperator

from zquantum.core.openfermion import save_qubit_operator_set, load_qubit_operator_set

from operators import concatenate_qubit_operator_lists

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

