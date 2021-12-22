import unittest

import numpy as np
from zquantum.core.circuits import Circuit
from zquantum.core.circuits._basis_gateset import RZRYCX
from zquantum.core.circuits._builtin_gates import RH, RX
from zquantum.core.decompositions._ryrzcx_decompositions import RXtoRZRY


class test_RZRYCX(unittest.TestCase):
    def setUp(self):
        self.basis = RZRYCX([RXtoRZRY()])
        self.gate_operation = RX(0.2)(2)
        self.invalid_gate_operation = RH(0.3)(0)
        self.circuit = Circuit([self.gate_operation])
        self.invalid_circuit = Circuit([self.invalid_gate_operation])
        self.targets = [
            ["RZ", (np.pi / 2,), (2,)],
            ["RY", ((0.2,),), (2,)],
            ["RZ", (-np.pi / 2,), (2,)],
        ]

    def test_decompose_operation(self):
        decomp_circuit = self.basis.decompose_operation(self.gate_operation)

        self.assertEqual(len(decomp_circuit.operations), 3)

        for operation, target in zip(decomp_circuit.operations, self.targets):
            self.assertEqual(operation.gate.name, target[0])
            self.assertEqual(operation.gate.params, target[1])
            self.assertEqual(operation.qubit_indices, target[2])

        self.assertRaises(
            RuntimeError, self.basis.decompose_operation, self.invalid_gate_operation
        )

    def test_decompose_circuit(self):
        decomp_circuit = self.basis.decompose_circuit(self.circuit)

        self.assertEqual(len(decomp_circuit.operations), 3)

        for operation, target in zip(decomp_circuit.operations, self.targets):
            self.assertEqual(operation.gate.name, target[0])
            self.assertEqual(operation.gate.params, target[1])
            self.assertEqual(operation.qubit_indices, target[2])

        self.assertRaises(
            RuntimeError, self.basis.decompose_circuit, self.invalid_circuit
        )

    def test_is_overcomplete(self):
        self.assertFalse(self.basis.is_overcomplete())
