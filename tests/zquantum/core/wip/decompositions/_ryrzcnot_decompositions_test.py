import unittest

import numpy as np
from zquantum.core.circuits import GPHASE, RX, RY, RZ, X
from zquantum.core.wip.decompositions._ryrzcnot_decompositions import RXtoRZRY, XtoRZRY


class TestDecompositionRXIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = RXtoRZRY()
        self.operation_to_decompose = RX(0.2)(2)
        self.targets = [
            RZ(-np.pi / 2)(2),
            RY(0.2)(2),
            RZ(np.pi / 2)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionXIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = XtoRZRY()
        self.operation_to_decompose = X(2)
        self.targets = [
            GPHASE(np.pi / 2)(2),
            RZ(-np.pi / 2)(2),
            RY(np.pi)(2),
            RZ(np.pi / 2)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)
