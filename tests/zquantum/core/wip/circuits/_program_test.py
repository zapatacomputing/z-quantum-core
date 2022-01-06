import unittest

import numpy as np
from zquantum.core.circuits import RX, RY, RZ, Circuit
from zquantum.core.wip.circuits import RZRYCNOT, Program
from zquantum.core.wip.decompositions._ryrzcnot_decompositions import RXtoRZRY


class TestProgram(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = Circuit([RX(0.2)(1), RX(0.5)(1)])
        self.basis = RZRYCNOT([RXtoRZRY()])
        self.program = Program()

    def test_append(self):
        self.program.append(self.circuit, self.basis)

        self.assertEqual(len(self.program.instructions), 1)

        targets = [
            RX(0.2)(1),
            RX(0.5)(1),
        ]

        for circuit, basis in self.program.instructions:
            for operation, target in zip(circuit.operations, targets):
                self.assertEqual(operation, target)

    def test_decompose(self):
        self.program.append(self.circuit, self.basis)

        self.program.decompose()

        targets = [
            RZ(np.pi / 2)(1),
            RY(0.2)(1),
            RZ(-np.pi / 2)(1),
            RZ(np.pi / 2)(1),
            RY(0.5)(1),
            RZ(-np.pi / 2)(1),
        ]

        for circuit, basis in self.program.instructions:
            for operation, target in zip(circuit.operations, targets):
                self.assertEqual(operation, target)
