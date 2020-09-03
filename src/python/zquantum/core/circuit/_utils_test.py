import unittest
import os
from openfermion import QubitOperator
from . import create_circuits_from_qubit_operator

from . import (
    Circuit,
    Gate,
    Qubit,
)

# from ._qubit import Qubit
# import pyquil
# import cirq
# import qiskit


class TestUtils(unittest.TestCase):
    
    def test_create_circuits_from_qubit_op(self):
        # Initialize target

        qubits = [Qubit(i) for i in range(0, 2)]

        gate_Z0 = Gate("Z", [qubits[0]])
        gate_X1 = Gate("X", [qubits[1]])

        gate_Y0 = Gate("Y", [qubits[0]])
        gate_Z1 = Gate("Z", [qubits[1]])
        
        circuit1 = Circuit()
        circuit1.qubits = qubits
        circuit1.gates = [gate_Z0, gate_X1]

        circuit2 = Circuit()
        circuit2.qubits = qubits
        circuit2.gates = [gate_Y0, gate_Z1]

        target_circuits_list = [circuit1, circuit2]

        # Given
        qubit_op = QubitOperator('Z0 X1') + QubitOperator('Y0 Z1')

        # When
        pauli_circuits = create_circuits_from_qubit_operator(qubit_op)
        
        # Then
        self.assertEqual(pauli_circuits[0].gates, target_circuits_list[0].gates)
        self.assertEqual(pauli_circuits[1].gates, target_circuits_list[1].gates)
        self.assertEqual(str(pauli_circuits[0].qubits), str(target_circuits_list[0].qubits))
        self.assertEqual(str(pauli_circuits[1].qubits), str(target_circuits_list[1].qubits))

