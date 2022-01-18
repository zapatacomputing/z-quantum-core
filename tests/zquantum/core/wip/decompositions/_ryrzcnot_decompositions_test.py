from unicodedata import decomposition
import unittest
import pytest

import numpy as np
from sympy import eye, simplify
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.core.measurement import ExpectationValues
from zquantum.core.interfaces.estimation import EstimationTask
from openfermion import IsingOperator
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.circuits import (
    Circuit,
    GPHASE,
    PHASE,
    RX,
    RY,
    RZ,
    U3,
    I,
    X,
    Y,
    Z,
    H,
    S,
    T,
    CNOT,
    CPHASE,
    CZ,
    SWAP,
    ISWAP,
)
from zquantum.core.wip.decompositions._ryrzcnot_decompositions import (
    PHASEtoRZRY,
    RXtoRZRY,
    U3toRZRY,
    ItoRZRY,
    XtoRZRY,
    YtoRZRY,
    ZtoRZRY,
    HtoRZRY,
    StoRZRY,
    TtoRZRY,
    CPHASEtoRZRYCNOT,
    CZtoRZRYCNOT,
    SWAPtoRZRYCNOT,
    ISWAPtoRZRYCNOT,
)

PARAMS = [
    (
        PHASEtoRZRY(),
        PHASE(np.pi)(1),
        [
            RZ(np.pi)(1),
            GPHASE(np.pi / 2)(1),
        ],
    ),
    (
        RXtoRZRY(),
        RX(0.2)(1),
        [
            RZ(-np.pi / 2)(1),
            RY(0.2)(1),
            RZ(np.pi / 2)(1),
        ],
    ),
    (
        U3toRZRY(),
        U3(0.2, 0.1, 0.3)(1),
        [
            RZ(0.1)(1),
            RY(0.2)(1),
            RZ(0.3)(1),
            GPHASE(0.2)(1),
        ],
    ),
    (
        ItoRZRY(),
        I(1),
        [
            RZ(0)(1),
            RY(0)(1),
            RZ(0)(1),
        ],
    ),
    (
        XtoRZRY(),
        X(1),
        [
            RZ(-np.pi / 2)(1),
            RY(np.pi)(1),
            RZ(np.pi / 2)(1),
            GPHASE(np.pi / 2)(1),
        ],
    ),
    (
        YtoRZRY(),
        Y(1),
        [
            RY(np.pi)(1),
            GPHASE(np.pi / 2)(1),
        ],
    ),
    (
        ZtoRZRY(),
        Z(1),
        [
            RZ(np.pi)(1),
            GPHASE(np.pi / 2)(1),
        ],
    ),
    (
        HtoRZRY(),
        H(1),
        [
            RY(np.pi / 2)(1),
            RZ(np.pi)(1),
            GPHASE(np.pi / 2)(1),
        ],
    ),
    (
        StoRZRY(),
        S(1),
        [
            RZ(np.pi / 2)(1),
            GPHASE(np.pi / 4)(1),
        ],
    ),
    (
        TtoRZRY(),
        T(1),
        [
            RZ(np.pi / 4)(1),
            GPHASE(np.pi / 8)(1),
        ],
    ),
    (
        CPHASEtoRZRYCNOT(),
        CPHASE(np.pi / 2)(0, 1),
        [
            RZ(np.pi / 4)(1),
            CNOT(0, 1),
            RZ(np.pi / 4)(0),
            RZ(-np.pi / 4)(1),
            CNOT(0, 1),
            GPHASE(np.pi / 8)(1),
        ],
    ),
    (
        CZtoRZRYCNOT(),
        CZ(0, 1),
        [
            RY(np.pi / 2)(1),
            RZ(np.pi)(1),
            GPHASE(np.pi / 2)(1),
            CNOT(0, 1),
            RY(np.pi / 2)(1),
            RZ(np.pi)(1),
            GPHASE(np.pi / 2)(1),
        ],
    ),
    (
        SWAPtoRZRYCNOT(),
        SWAP(0, 1),
        [
            CNOT(0, 1),
            CNOT(1, 0),
            CNOT(0, 1),
        ],
    ),
    (
        ISWAPtoRZRYCNOT(),
        ISWAP(0, 1),
        [
            RZ(np.pi / 2)(0),
            RZ(np.pi / 2)(1),
            RY(np.pi / 2)(0),
            RZ(np.pi)(0),
            CNOT(0, 1),
            CNOT(1, 0),
            RY(np.pi / 2)(1),
            RZ(np.pi)(1),
            GPHASE(-np.pi / 2)(1),
        ],
    ),
]


class TestDecompositions:
    @pytest.mark.parametrize("decomp_rule, operation_to_decompose, target", PARAMS)
    def test_predicate(self, decomp_rule, operation_to_decompose, target):
        assert decomp_rule.predicate(operation_to_decompose)

    @pytest.mark.parametrize("decomp_rule, operation_to_decompose, target", PARAMS)
    def test_production(self, decomp_rule, operation_to_decompose, target):
        decomposition = decomp_rule.production(operation_to_decompose)

        prod = eye(4)
        for operation, target in zip(decomposition, target):
            assert operation == target
            prod *= operation.lifted_matrix(2)

        assert (
            simplify(operation_to_decompose.lifted_matrix(2) - prod).norm() < 10 ** -12
        )
