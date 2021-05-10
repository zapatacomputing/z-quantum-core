import unittest
from math import pi

import qiskit
from sympy import Symbol
from zquantum.core.circuit._gate import Gate
from zquantum.core.circuit._gateset import COMMON_GATES
from zquantum.core.circuit._qubit import Qubit


class TestGate(unittest.TestCase):
    def setUp(self):
        self.two_qubit_gates = ["CNOT", "CZ", "CPHASE", "SWAP", "ISWAP"]
        self.one_parameter_gates = ["PHASE", "Rx", "Ry", "Rz", "CPHASE"]

    def create_gate(self, gate_name, qubit_indices=[0, 1], params=None):
        if gate_name in self.two_qubit_gates:
            qubit_list = [Qubit(qubit_indices[0]), Qubit(qubit_indices[1])]
        else:
            qubit_list = [Qubit(qubit_indices[0])]
        if params is None:
            params = []
            if gate_name in self.one_parameter_gates:
                params = [1.0]
        gate = Gate(gate_name, qubits=qubit_list, params=params)
        return gate

    def create_gate_with_symbolic_params(self, gate_name, qubit_indices=[0, 1]):
        if gate_name in self.two_qubit_gates:
            qubit_list = [Qubit(qubit_indices[0]), Qubit(qubit_indices[1])]
        else:
            qubit_list = [Qubit(qubit_indices[0])]
        params = []
        if gate_name in self.one_parameter_gates:
            params = [Symbol("theta_0")]
        gate = Gate(gate_name, qubits=qubit_list, params=params)
        return gate, params

    def test_evaluate_works_with_regular_gate(self):

        for gate_name in self.one_parameter_gates:
            # Given
            gate = self.create_gate(gate_name)
            symbols_map = [(Symbol("theta_0"), 1.0)]

            # When
            evaluated_regular_gate = gate.evaluate(symbols_map)

            # Then
            self.assertEqual(evaluated_regular_gate, gate)

    def test_evaluate_works_with_symbolic_gate(self):

        for gate_name in self.one_parameter_gates:
            # Given
            param_value = 1.0
            gate = self.create_gate(gate_name, params=[param_value])
            symbolic_gate, params = self.create_gate_with_symbolic_params(gate_name)
            symbols_map = [(params[0], param_value)]

            # When
            evaluated_symbolic_gate = symbolic_gate.evaluate(symbols_map)

            # Then
            self.assertEqual(gate, evaluated_symbolic_gate)
            # Check if the params of the initial gate has not been overwritten
            self.assertEqual(symbolic_gate.params[0], symbols_map[0][0])

            # Given
            symbols_map = [("x", 1.0)]

            # When
            evaluated_symbolic_gate = symbolic_gate.evaluate(symbols_map)

            # Then
            self.assertEqual(evaluated_symbolic_gate, symbolic_gate)

    def test_symbolic_params(self):

        # Given
        params = [0.5, Symbol("theta_0"), Symbol("theta_0") + 2 * Symbol("theta_1")]
        target_symbolic_params = [
            [],
            [Symbol("theta_0")],
            [Symbol("theta_0"), Symbol("theta_1")],
        ]

        for param, target_params in zip(params, target_symbolic_params):
            # Given
            qubit_list = [Qubit(0)]
            gate = Gate("Rx", qubits=qubit_list, params=[param])

            # When
            symbolic_params = gate.symbolic_params

            # Then
            self.assertEqual(symbolic_params, target_params)

    def test_dict_io(self):
        for gate_name in COMMON_GATES:
            # Given
            gate = self.create_gate(gate_name)

            # When
            gate_dict = gate.to_dict()
            recreated_gate = Gate.from_dict(gate_dict)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_pyquil_io(self):
        for gate_name in COMMON_GATES:
            # Given
            gate = self.create_gate(gate_name)

            # When
            pyquil_gate = gate.to_pyquil()
            qubits = [
                Qubit.from_pyquil(pyquil_qubit) for pyquil_qubit in pyquil_gate.qubits
            ]

            recreated_gate = Gate.from_pyquil(pyquil_gate, qubits)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_cirq_io(self):
        for gate_name in COMMON_GATES:
            # Given
            gate = self.create_gate(gate_name)

            # When
            cirq_gate = gate.to_cirq()
            qubits = [
                Qubit.from_cirq(cirq_qubit, cirq_qubit.x)
                for cirq_qubit in cirq_gate.qubits
            ]
            recreated_gate = Gate.from_cirq(cirq_gate, qubits)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_qiskit_io(self):
        for gate_name in COMMON_GATES:
            # Given
            gate = self.create_gate(gate_name)
            qreg = qiskit.QuantumRegister(2, "q")
            creg = qiskit.ClassicalRegister(2, "c")

            # When
            qiskit_gate, qreg, creg = gate.to_qiskit(qreg, creg)
            recreated_gate = Gate.from_qiskit(qiskit_gate, gate.qubits)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_dict_io_for_symbolic_parameters(self):
        for gate_name in self.one_parameter_gates:
            # Given
            gate, _ = self.create_gate_with_symbolic_params(gate_name)

            # When
            gate_dict = gate.to_dict()
            gate_dict_serialized = gate.to_dict(serialize_params=True)
            recreated_gate = Gate.from_dict(gate_dict)
            recreated_gate_from_serialized = Gate.from_dict(gate_dict_serialized)

            # Then
            self.assertEqual(gate, recreated_gate)
            self.assertEqual(gate, recreated_gate_from_serialized)

    def test_pyquil_io_for_symbolic_parameters(self):
        for gate_name in self.one_parameter_gates:
            # Given
            gate, _ = self.create_gate_with_symbolic_params(gate_name)

            # When
            pyquil_gate = gate.to_pyquil()
            qubits = [
                Qubit.from_pyquil(pyquil_qubit) for pyquil_qubit in pyquil_gate.qubits
            ]
            recreated_gate = Gate.from_pyquil(pyquil_gate, qubits)

            # Then
            self.assertEqual(gate, recreated_gate)

    def test_cirq_io_for_symbolic_parameters(self):
        for gate_name in self.one_parameter_gates:
            for param_value in [-pi, 0, pi / 2, pi, 2 * pi, 0.38553]:
                # Given
                gate, params = self.create_gate_with_symbolic_params(gate_name)
                symbols_map = []
                for param in params:
                    symbols_map.append((param, param_value))

                # When
                cirq_gate = gate.to_cirq()
                qubits = [
                    Qubit.from_cirq(cirq_qubit, cirq_qubit.x)
                    for cirq_qubit in cirq_gate.qubits
                ]
                recreated_gate = Gate.from_cirq(cirq_gate, qubits)

                recreated_gate = Gate.from_cirq(cirq_gate, qubits)
                gate_evaluated = gate.evaluate(symbols_map)
                recreated_gate_evaluated = recreated_gate.evaluate(symbols_map)

                # Then
                # There were numerical & sympy related issues when comparing gates directly, so in this case we compare the evaluated forms of the gates.
                self.assertEqual(gate_evaluated, recreated_gate_evaluated)

    def test_qiskit_io_for_symbolic_parameters(self):
        for gate_name in self.one_parameter_gates:
            # Given
            gate, params = self.create_gate_with_symbolic_params(gate_name)
            qreg = qiskit.QuantumRegister(2, "q")
            creg = qiskit.ClassicalRegister(2, "c")

            # When
            qiskit_gate, qreg, creg = gate.to_qiskit(qreg, creg)
            recreated_gate = Gate.from_qiskit(qiskit_gate, gate.qubits)

            # Then
            self.assertEqual(gate, recreated_gate)
