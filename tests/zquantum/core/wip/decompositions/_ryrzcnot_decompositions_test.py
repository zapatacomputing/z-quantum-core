import unittest
import pytest

import numpy as np
from zquantum.core.circuits import (
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
        PHASE(np.pi)(2),
        [
            RZ(np.pi)(2),
            GPHASE(np.pi / 2)(2),
        ],
    ),
    (
        RXtoRZRY(),
        RX(0.2)(2),
        [
            RZ(-np.pi / 2)(2),
            RY(0.2)(2),
            RZ(np.pi / 2)(2),
        ],
    ),
    (
        U3toRZRY(),
        U3(0.1, 0.2, 0.3)(2),
        [
            RZ(0.1)(2),
            RY(0.2)(2),
            RZ(0.3)(2),
            GPHASE(0.2)(2),
        ],
    ),
    (
        ItoRZRY(),
        I(2),
        [
            RZ(0)(2),
            RY(0)(2),
            RZ(0)(2),
        ],
    ),
    (
        XtoRZRY(),
        X(2),
        [
            RZ(-np.pi / 2)(2),
            RY(np.pi)(2),
            RZ(np.pi / 2)(2),
            GPHASE(np.pi / 2)(2),
        ],
    ),
    (
        YtoRZRY(),
        Y(2),
        [
            RY(np.pi)(2),
            GPHASE(np.pi / 2)(2),
        ],
    ),
    (
        ZtoRZRY(),
        Z(2),
        [
            RZ(np.pi)(2),
            GPHASE(np.pi / 2)(2),
        ],
    ),
    (
        HtoRZRY(),
        H(2),
        [
            RY(np.pi / 2)(2),
            RZ(np.pi)(2),
            GPHASE(np.pi / 2)(2),
        ],
    ),
    (
        StoRZRY(),
        S(2),
        [
            RZ(np.pi / 2)(2),
            GPHASE(np.pi / 4)(2),
        ],
    ),
    (
        TtoRZRY(),
        T(2),
        [
            RZ(np.pi / 4)(2),
            GPHASE(np.pi / 8)(2),
        ],
    ),
    (
        CPHASEtoRZRYCNOT(),
        CPHASE(np.pi)(1, 2),
        [
            RZ(-np.pi / 2)(2),
            CNOT(1, 2),
            RZ(-np.pi / 2),
            CNOT(1, 2),
            RZ(np.pi),
            GPHASE(np.pi / 2),
        ],
    ),
    (
        CZtoRZRYCNOT(),
        CZ(1, 2),
        [
            RY(np.pi / 2)(2),
            RZ(np.pi)(2),
            GPHASE(np.pi / 2)(2),
            CNOT(1, 2),
            RY(np.pi / 2)(2),
            RZ(np.pi)(2),
            GPHASE(np.pi / 2)(2),
        ],
    ),
    (
        SWAPtoRZRYCNOT(),
        SWAP(1, 2),
        [
            CNOT(1, 2),
            CNOT(2, 1),
            CNOT(1, 2),
        ],
    ),
    (
        ISWAPtoRZRYCNOT(),
        ISWAP(1, 2),
        [
            CNOT(1, 2),
            CNOT(2, 1),
            CNOT(1, 2),
            RZ(-np.pi / 2)(1),
            RZ(np.pi / 2)(2),
            RY(-np.pi / 2)(2),
            CNOT(1, 2),
            RY(-np.pi / 2)(2),
        ],
    ),
]


class TestDecompositions:
    @pytest.mark.parametrize("decomp_rule, operation_to_decompose, targets", PARAMS)
    def test_predicate(self, decomp_rule, operation_to_decompose, targets):
        assert decomp_rule.predicate(operation_to_decompose)

    @pytest.mark.parametrize("decomp_rule, operation_to_decompose, targets", PARAMS)
    def test_production(self, decomp_rule, operation_to_decompose, targets):
        prod = decomp_rule.production(operation_to_decompose)

        for operation, target in zip(prod, targets):
            assert operation == target
