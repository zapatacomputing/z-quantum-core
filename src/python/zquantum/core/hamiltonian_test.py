from .hamiltonian import is_comeasureable, group_comeasureable_terms_greedy, compute_group_variances
from .measurement import ExpectationValues
import numpy as np
import pytest
from openfermion.ops import QubitOperator


@pytest.mark.parametrize(
    "term1,term2,expected_result",
    [
        (((0, "Y"),), ((0, "X"),), False),
        (((0, "Y"),), ((1, "X"),), True),
        (((0, "Y"), (1, "X")), (), True),
    ],
)
def test_is_comeasureable(term1, term2, expected_result):
    assert is_comeasureable(term1, term2) == expected_result


@pytest.mark.parametrize(
    "qubit_operator,expected_groups",
    [
        (
            QubitOperator("[Z0 Z1] + [X0 X1] + [Z0] + [X0]"),
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
        ),
    ],
)
def test_group_comeasureable_terms_greedy(qubit_operator, expected_groups):
    groups = group_comeasureable_terms_greedy(qubit_operator)
    assert groups == expected_groups


@pytest.mark.parametrize(
    "qubit_operator,sort_terms,expected_groups",
    [
        (
            QubitOperator("[Z0 Z1] + [X0 X1] + [Z0] + [X0]"),
            True,
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
        ),
        (
            QubitOperator("[X0] + 2 [X0 Y1] + 3 [X0 Z1]"),
            True,
            [QubitOperator("[X0] + 3 [X0 Z1]"), QubitOperator("2 [X0 Y1]")],
        ),
        (
            QubitOperator("[X0] + 2 [X0 Y1] + 3 [X0 Z1]"),
            False,
            [QubitOperator("[X0] + 2 [X0 Y1]"), QubitOperator("3 [X0 Z1]")],
        ),
    ],
)
def test_group_comeasureable_terms_greedy_sorted(
    qubit_operator, sort_terms, expected_groups
):
    groups = group_comeasureable_terms_greedy(qubit_operator, sort_terms=sort_terms)
    assert groups == expected_groups

@pytest.mark.parametrize(
    "groups, expecval, variances",
    [
        (
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")], None, np.array([ 2., 2. ]),
        ),
        (
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")], ExpectationValues([0., 0., 0., 0.]), np.array([ 2., 2. ]),
        ),
    ],
)
def test_compute_group_variances(groups, expecval, variances):
    test_variances = compute_group_variances(groups, expecval)
    assert np.allclose(test_variances, variances)
