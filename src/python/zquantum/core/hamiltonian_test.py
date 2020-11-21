from .hamiltonian import is_comeasureable, group_comeasureable_terms_greedy, compute_group_variances, get_expectation_values_from_rdms
from .measurement import ExpectationValues
import numpy as np
import math
import pytest
from openfermion.ops import QubitOperator, InteractionRDM

h2_hamiltonian = QubitOperator("""-0.0420789769629383 [] +
-0.04475014401986127 [X0 X1 Y2 Y3] +
0.04475014401986127 [X0 Y1 Y2 X3] +
0.04475014401986127 [Y0 X1 X2 Y3] +
-0.04475014401986127 [Y0 Y1 X2 X3] +
0.17771287459806312 [Z0] +
0.1705973832722407 [Z0 Z1] +
0.12293305054268083 [Z0 Z2] +
0.1676831945625421 [Z0 Z3] +
0.17771287459806312 [Z1] +
0.1676831945625421 [Z1 Z2] +
0.12293305054268083 [Z1 Z3] +
-0.24274280496459985 [Z2] +
0.17627640802761105 [Z2 Z3] +
-0.24274280496459985 [Z3]""")

rdm1 = np.array([[0.98904311, 0.        , 0.        , 0.        ],
       [0.        , 0.98904311, 0.        , 0.        ],
       [0.        , 0.        , 0.01095689, 0.        ],
       [0.        , 0.        , 0.        , 0.01095689]])

rdm2 = np.array([[[[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]],

        [[ 0.        , -0.98904311,  0.        , -0.        ],
         [ 0.98904311,  0.        ,  0.        ,  0.        ],
         [ 0.        , -0.        ,  0.        ,  0.10410015],
         [ 0.        ,  0.        , -0.10410015,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]],

        [[ 0.        , -0.        ,  0.        , -0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        , -0.        ,  0.        , -0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]]],


       [[[ 0.        ,  0.98904311,  0.        ,  0.        ],
         [-0.98904311,  0.        , -0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        , -0.10410015],
         [-0.        ,  0.        ,  0.10410015,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ],
         [-0.        ,  0.        , -0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [-0.        ,  0.        , -0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]]],


       [[[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]],

        [[ 0.        , -0.        ,  0.        , -0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        , -0.        ,  0.        , -0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]],

        [[ 0.        ,  0.10410015,  0.        , -0.        ],
         [-0.10410015,  0.        ,  0.        ,  0.        ],
         [ 0.        , -0.        ,  0.        , -0.01095689],
         [ 0.        ,  0.        ,  0.01095689,  0.        ]]],


       [[[ 0.        ,  0.        ,  0.        ,  0.        ],
         [-0.        ,  0.        , -0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [-0.        ,  0.        , -0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]],

        [[ 0.        , -0.10410015,  0.        ,  0.        ],
         [ 0.10410015,  0.        , -0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.01095689],
         [-0.        ,  0.        , -0.01095689,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ]]]])

rdms = InteractionRDM(rdm1, rdm2)

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
    "interactionrdm, qubitoperator, sort_terms",
    [
        (
            rdms, h2_hamiltonian, False,
        ),
    ],
)

def test_get_expectation_values_from_rdms(interactionrdm,
    qubitoperator, sort_terms
):
    expecval = get_expectation_values_from_rdms(interactionrdm, qubitoperator, sort_terms)
    energy_test = np.sum(np.array(list(qubitoperator.terms.values())) * expecval.values)
    energy_ref = np.real(interactionrdm.expectation(qubitoperator))
    assert math.isclose(energy_test, energy_ref)

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
