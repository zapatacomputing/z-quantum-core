import unittest
import os
from ._gate import Gate
from ._gateset import COMMON_GATES, UNIQUE_GATES
from ._qubit import Qubit
from sympy import Symbol
import qiskit
from math import pi


class TestGate(unittest.TestCase):
    def setUp(self):
        self.two_qubit_gates = ["CNOT", "CZ", "CPHASE", "SWAP"]
        self.one_parameter_gates = ["PHASE", "Rx", "Ry", "Rz", "CPHASE"]

    def create_gate(self, gate_name, params=None):
        if gate_name in self.two_qubit_gates:
            qubit_list = [Qubit(0), Qubit(1)]
        else:
            qubit_list = [Qubit(0)]
        if params is None:
            params = []
            if gate_name in self.one_parameter_gates:
                params = [1.0]
        gate = Gate(gate_name, qubits=qubit_list, params=params)
        return gate, qubit_list

    def create_gate_with_symbolic_params(self, gate_name):
        if gate_name in self.two_qubit_gates:
            qubit_list = [Qubit(0), Qubit(1)]
        else:
            qubit_list = [Qubit(0)]
        params = []
        if gate_name in self.one_parameter_gates:
            params = [Symbol("theta_0")]
        gate = Gate(gate_name, qubits=qubit_list, params=params)
        return gate, qubit_list, params

    def test_create_evaluated_gate(self):
        # Given
        gate, _ = self.create_gate("Rx")
        symbolic_gate, _, params = self.create_gate_with_symbolic_params("Rx")
        symbols_map = [(params[0], 1.0)]

        # When
        evaluated_symbolic_gate = symbolic_gate.create_evaluated_gate(symbols_map)
        evaluated_regular_gate = gate.create_evaluated_gate(symbols_map)

        # Then
        self.assertEqual(gate, evaluated_symbolic_gate)
        # Check if the params of the initial gate has not been overwritten
        self.assertEqual(symbolic_gate.params[0], symbols_map[0][0])
        # Check if evaluating a regular gate returns the exact same gate
        self.assertEqual(evaluated_regular_gate, gate)

        # Given
        symbols_map = [("x", 1.0)]

        # When
        evaluated_symbolic_gate = symbolic_gate.create_evaluated_gate(symbols_map)

        # Then
        self.assertEqual(evaluated_symbolic_gate, symbolic_gate)

    def test_dict_io(self):
        for gate_name in COMMON_GATES:
            # Given
            gate, _ = self.create_gate(gate_name)

            # When
            gate_dict = gate.to_dict()
            recreated_gate = Gate.from_dict(gate_dict)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_pyquil_io(self):
        for gate_name in COMMON_GATES:
            # Given
            gate, qubit_list = self.create_gate(gate_name)

            # When
            pyquil_gate = gate.to_pyquil()
            recreated_gate = Gate.from_pyquil(pyquil_gate, qubit_list)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_cirq_io(self):
        for gate_name in COMMON_GATES:
            # Given
            gate, qubit_list = self.create_gate(gate_name)

            # When
            cirq_gate = gate.to_cirq()
            recreated_gate = Gate.from_cirq(cirq_gate, qubit_list)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_qiskit_io(self):
        for gate_name in COMMON_GATES:
            # Given
            gate, qubit_list = self.create_gate(gate_name)
            qreg = qiskit.QuantumRegister(2, "q")
            cphase_targets = [
                ("Rx", [pi / 2]),
                ("Ry", [pi - 0.5]),
                ("CZ", None),
                ("Ry", [-pi + 0.5]),
                ("Rx", [-pi]),
                ("CZ", None),
                ("Rx", [pi / 2]),
                ("Rz", [0.5]),
            ]

            # When
            if gate_name == "CPHASE":
                cphase_decomposition = gate.to_qiskit(qreg=qreg)
                for i in range(int(len(cphase_decomposition) / 3)):
                    current_gate = cphase_decomposition[3 * i]
                    current_qreg = cphase_decomposition[3 * i + 1]
                    target_gate, qubit_list = self.create_gate(
                        cphase_targets[i][0], params=cphase_targets[i][1]
                    )
                    recreated_current_gate = Gate.from_qiskit(current_gate, qubit_list)
                    self.assertEqual(target_gate, recreated_current_gate)
                continue

            else:
                qiskit_gate, qreg, creg = gate.to_qiskit(qreg=qreg)
                recreated_gate = Gate.from_qiskit(qiskit_gate, qubit_list)

            # Then
            if gate_name == "PHASE":
                rz_gate, _ = self.create_gate("Rz")
                self.assertEqual(rz_gate, recreated_gate)
            else:
                self.assertEqual(gate, recreated_gate)

    def test_dict_io_for_symbolic_parameters(self):
        for gate_name in self.one_parameter_gates:
            # Given
            gate, _, _ = self.create_gate_with_symbolic_params(gate_name)

            # When
            gate_dict = gate.to_dict()
            recreated_gate = Gate.from_dict(gate_dict)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_pyquil_io_for_symbolic_parameters(self):
        for gate_name in self.one_parameter_gates:
            # Given
            gate, qubit_list, _ = self.create_gate_with_symbolic_params(gate_name)

            # When
            pyquil_gate = gate.to_pyquil()
            recreated_gate = Gate.from_pyquil(pyquil_gate, qubit_list)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_cirq_io_for_symbolic_parameters(self):
        for gate_name in self.one_parameter_gates:
            # Given
            gate, qubit_list, params = self.create_gate_with_symbolic_params(gate_name)
            symbols_map = []
            for param in params:
                symbols_map.append((param, 0.38553))

            # When
            cirq_gate = gate.to_cirq()
            recreated_gate = Gate.from_cirq(cirq_gate, qubit_list)
            gate_evaluated = gate.create_evaluated_gate(symbols_map)
            recreated_gate_evaluated = recreated_gate.create_evaluated_gate(symbols_map)

            # Then
            # There were numerical & sympy related issues when comparing gates directly, so in this case we compare the evaluated forms of the gates.
            self.assertEqual(gate_evaluated, recreated_gate_evaluated)

    def test_qiskit_io_for_symbolic_parameters(self):
        pass
