import unittest

import numpy as np
from zquantum.core.circuits import GPHASE, PHASE, RX, RY, RZ, U3, I, X, Y, Z, H, S, T, CNOT, CPHASE, CZ, SWAP, ISWAP
from zquantum.core.wip.decompositions._ryrzcnot_decompositions import PHASEtoRZRY, RXtoRZRY, U3toRZRY, ItoRZRY, XtoRZRY, YtoRZRY, ZtoRZRY, HtoRZRY, StoRZRY, TtoRZRY, CPHASEtoRZRYCNOT, CZtoRZRYCNOT, SWAPtoRZRYCNOT, ISWAPtoRZRYCNOT


class TestDecompositionPHASEIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = PHASEtoRZRY()
        self.operation_to_decompose = PHASE(np.pi)(2)
        self.targets = [
            RZ(np.pi)(2),
            GPHASE(np.pi / 2)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


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


class TestDecompositionU3IntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = U3toRZRY()
        self.operation_to_decompose = U3(0.1, 0.2, 0.3)(2)
        self.targets = [
            RZ(0.1)(2),
            RY(0.2)(2),
            RZ(0.3)(2),
            GPHASE(0.2)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionIIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = ItoRZRY()
        self.operation_to_decompose = I(2)
        self.targets = [
            RZ(0)(2),
            RY(0)(2),
            RZ(0)(2),
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


class TestDecompositionYIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = YtoRZRY()
        self.operation_to_decompose = Y(2)
        self.targets = [
            GPHASE(np.pi/2)(2),
            RY(np.pi)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionZIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = ZtoRZRY()
        self.operation_to_decompose = Z(2)
        self.targets = [
            GPHASE(np.pi/2)(2),
            RZ(np.pi)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionHIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = HtoRZRY()
        self.operation_to_decompose = H(2)
        self.targets = [
            GPHASE(np.pi/2)(2),
            RY(np.pi/2)(2),
            RZ(np.pi)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionSIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = StoRZRY()
        self.operation_to_decompose = S(2)
        self.targets = [
            RZ(np.pi / 2)(2),
            GPHASE(np.pi / 4)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionTIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = TtoRZRY()
        self.operation_to_decompose = T(2)
        self.targets = [
            RZ(np.pi / 4)(2),
            GPHASE(np.pi / 8)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionCPHASEIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = CPHASEtoRZRYCNOT()
        self.operation_to_decompose = CPHASE(np.pi)(1, 2)
        self.targets = [
            RZ(-np.pi / 2)(2),
            CNOT(1, 2),
            RZ(-np.pi / 2),
            CNOT(1, 2),
            RZ(np.pi),
            GPHASE(np.pi / 2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionCZIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = CZtoRZRYCNOT()
        self.operation_to_decompose = CZ(1, 2)
        self.targets = [
            GPHASE(np.pi/2)(2),
            RY(np.pi/2)(2),
            RZ(np.pi)(2),
            CNOT(1, 2),
            GPHASE(np.pi/2)(2),
            RY(np.pi/2)(2),
            RZ(np.pi)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionSWAPIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = SWAPtoRZRYCNOT()
        self.operation_to_decompose = SWAP(1, 2)
        self.targets = [
            CNOT(1, 2),
            CNOT(2, 1),
            CNOT(1, 2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)


class TestDecompositionISWAPIntoRYRZCNOT(unittest.TestCase):
    def setUp(self) -> None:
        self.decomp_rule = ISWAPtoRZRYCNOT()
        self.operation_to_decompose = ISWAP(1, 2)
        self.targets = [
            CNOT(1, 2),
            CNOT(2, 1),
            CNOT(1, 2),
            RZ(-np.pi / 2)(1),
            RZ(np.pi / 2)(2),
            RY(-np.pi / 2)(2),
            CNOT(1, 2),
            RY(-np.pi / 2)(2),
        ]

    def test_predicate(self):
        assert self.decomp_rule.predicate(self.operation_to_decompose)

    def test_production(self):
        prod = self.decomp_rule.production(self.operation_to_decompose)

        for operation, target in zip(prod, self.targets):
            self.assertEqual(operation, target)
