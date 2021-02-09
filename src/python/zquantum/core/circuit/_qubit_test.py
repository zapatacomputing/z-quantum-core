import unittest
import os
from ._qubit import Qubit
import pyquil
import cirq
import qiskit


class TestQubit(unittest.TestCase):
    def test_dict_io(self):
        # Given
        qubit = Qubit(0)

        # When
        qubit_dict = qubit.to_dict()
        recreated_qubit = Qubit.from_dict(qubit_dict)

        # Then
        self.assertEqual(str(recreated_qubit), str(qubit))

    def test_from_pyquil(self):
        # Given
        pyquil_qubit = pyquil.quilatom.Qubit(0)
        target_qubit_dict = {"index": 0, "info": {"label": "pyquil"}}

        # When
        recreated_qubit = Qubit.from_pyquil(pyquil_qubit)

        # Then
        self.assertDictEqual(recreated_qubit.to_dict(), target_qubit_dict)

    def test_from_cirq(self):
        # Given
        index = 5
        cirq_grid_qubit = cirq.GridQubit(1, 2)
        cirq_line_qubit = cirq.LineQubit(3)

        target_grid_qubit_dict = {
            "index": 5,
            "info": {"label": "cirq", "QubitType": "GridQubit", "QubitKey": (1, 2)},
        }
        target_line_qubit_dict = {
            "index": 5,
            "info": {"label": "cirq", "QubitType": "LineQubit", "QubitKey": 3},
        }

        # When
        recreated_grid_qubit = Qubit.from_cirq(cirq_grid_qubit, index)
        recreated_line_qubit = Qubit.from_cirq(cirq_line_qubit, index)

        # Then
        self.assertDictEqual(recreated_grid_qubit.to_dict(), target_grid_qubit_dict)
        self.assertDictEqual(recreated_line_qubit.to_dict(), target_line_qubit_dict)

    def test_from_qiskit(self):
        # Given
        register = qiskit.QuantumRegister(3)
        index = 1
        qubit = qiskit.circuit.Qubit(register, index)
        target_qubit_dict = {
            "index": index,
            "info": {"label": "qiskit", "num": index},
        }

        # When
        recreated_qubit = Qubit.from_qiskit(qubit, index)

        # Then
        self.assertDictEqual(recreated_qubit.to_dict(), target_qubit_dict)
