import unittest

import numpy as np
from zquantum.core.circuits import RX, RY, RZ
from zquantum.core.decompositions._ryrzcnot_decompositions import RXtoRZRY


class TestDecompositionIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = RXtoRZRY()
        self.operation_to_decompose = RX(0.2)(2)
        self.targets = [
            RZ(np.pi / 2)(2),
            RY(0.2)(2),
            RZ(-np.pi / 2)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)
