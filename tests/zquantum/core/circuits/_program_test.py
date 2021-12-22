import unittest

import numpy as np
from zquantum.core.circuits import RX, RZRYCX, Circuit, Program
from zquantum.core.decompositions._ryrzcx_decompositions import RXtoRZRY


class TestProgram(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = Circuit([RX(0.2)(1), RX(0.5)(1)])
        self.basis = RZRYCX([RXtoRZRY()])
        self.program = Program()

    def test_append(self):
        self.program.append(self.circuit, self.basis)

        self.assertEqual(len(self.program.instructions), 1)

        target = [
            ["RX", (0.2,), (1,)],
            ["RX", (0.5,), (1,)],
        ]

        for circuit, basis in self.program.instructions:
            for operation, target in zip(circuit.operations, target):
                self.assertEqual(operation.gate.name, target[0])
                self.assertEqual(operation.gate.params, target[1])
                self.assertEqual(operation.qubit_indices, target[2])

    def test_decompose(self):
        self.program.append(self.circuit, self.basis)

        self.program.decompose()

        target = [
            ["RZ", (np.pi / 2,), (1,)],
            # TODO: why is circuit params now tuple of tuples?
            ["RY", ((0.2,),), (1,)],
            ["RZ", (-np.pi / 2,), (1,)],
            ["RZ", (np.pi / 2,), (1,)],
            ["RY", ((0.5,),), (1,)],
            ["RZ", (-np.pi / 2,), (1,)],
        ]

        for circuit, basis in self.program.instructions:
            for operation, target in zip(circuit.operations, target):
                self.assertEqual(operation.gate.name, target[0])
                self.assertEqual(operation.gate.params, target[1])
                self.assertEqual(operation.qubit_indices, target[2])
