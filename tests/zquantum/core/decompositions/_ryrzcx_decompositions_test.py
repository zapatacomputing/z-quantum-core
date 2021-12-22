import unittest

import numpy as np
from zquantum.core.circuits import RX
from zquantum.core.decompositions._ryrzcx_decompositions import RXtoRZRY


class TestDecompositionIntoRYrZcX(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = RXtoRZRY()
        self.operation_to_decompose = RX(0.2)(2)
        self.targets = [
            ["RZ", (np.pi / 2,), (2,)],
            ["RY", ((0.2,),), (2,)],
            ["RZ", (-np.pi / 2,), (2,)],
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        for operation, target in zip(
            self.decomp_rule.production(self.operation_to_decompose), self.targets
        ):
            assert operation.gate.name == target[0]
            assert operation.gate.params == target[1]
            assert operation.qubit_indices == target[2]
