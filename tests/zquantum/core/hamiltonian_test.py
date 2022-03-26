import math

import numpy as np
import pytest
from zquantum.core.hamiltonian import (
    compute_group_variances,
    estimate_nmeas_for_frames,
    get_expectation_values_from_rdms,
    get_expectation_values_from_rdms_for_qubitoperator_list,
    group_comeasureable_terms_greedy,
    is_comeasureable,
    reorder_fermionic_modes,
)
from zquantum.core.measurement import ExpectationValues
from zquantum.core.openfermion import (
    FermionOperator,
    InteractionRDM,
    QubitOperator,
    eigenspectrum,
    get_interaction_operator,
    jordan_wigner,
)

h2_hamiltonian = QubitOperator(
    """-0.0420789769629383 [] +
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
-0.24274280496459985 [Z3]"""
)

h2_hamiltonian_grouped = [
    QubitOperator("-0.0420789769629383 []"),
    QubitOperator("-0.04475014401986127 [X0 X1 Y2 Y3]"),
    QubitOperator("0.04475014401986127 [X0 Y1 Y2 X3]"),
    QubitOperator("0.04475014401986127 [Y0 X1 X2 Y3]"),
    QubitOperator("-0.04475014401986127 [Y0 Y1 X2 X3]"),
    QubitOperator(
        """0.17771287459806312 [Z0] +
         0.1705973832722407 [Z0 Z1] +
         0.12293305054268083 [Z0 Z2] +
         0.1676831945625421 [Z0 Z3] +
         0.17771287459806312 [Z1] +
         0.1676831945625421 [Z1 Z2] +
         0.12293305054268083 [Z1 Z3] +
         -0.24274280496459985 [Z2] +
         0.17627640802761105 [Z2 Z3] +
         -0.24274280496459985 [Z3]"""
    ),
]

rdm1 = np.array(
    [
        [0.98904311, 0.0, 0.0, 0.0],
        [0.0, 0.98904311, 0.0, 0.0],
        [0.0, 0.0, 0.01095689, 0.0],
        [0.0, 0.0, 0.0, 0.01095689],
    ]
)

rdm2 = np.array(
    [
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.98904311, 0.0, -0.0],
                [0.98904311, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, 0.10410015],
                [0.0, 0.0, -0.10410015, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.98904311, 0.0, 0.0],
                [-0.98904311, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, -0.10410015],
                [-0.0, 0.0, 0.10410015, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.10410015, 0.0, -0.0],
                [-0.10410015, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, -0.01095689],
                [0.0, 0.0, 0.01095689, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.10410015, 0.0, 0.0],
                [0.10410015, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.01095689],
                [-0.0, 0.0, -0.01095689, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
    ]
)

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
            QubitOperator("2[Z0 Z1] + [X0 X1] + [Z0] + [X0]"),
            [QubitOperator("2[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
        ),
        (
            QubitOperator("[Z0 Z1] + [X0 X1] + [Z0] + []"),
            [
                QubitOperator("[Z0 Z1] + [Z0]"),
                QubitOperator("[X0 X1]"),
                QubitOperator("[]"),
            ],
        ),
        (
            QubitOperator("-3[]"),
            [QubitOperator("-3[]")],
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
            rdms,
            h2_hamiltonian,
            False,
        ),
        (
            rdms,
            h2_hamiltonian,
            True,
        ),
    ],
)
def test_get_expectation_values_from_rdms(interactionrdm, qubitoperator, sort_terms):
    expecval = get_expectation_values_from_rdms(
        interactionrdm, qubitoperator, sort_terms
    )
    assert len(expecval.values) == len(qubitoperator.terms)
    if not sort_terms:
        coeff = np.array(list(qubitoperator.terms.values()))
    else:
        terms_iterator = sorted(
            qubitoperator.terms.items(), key=lambda x: abs(x[1]), reverse=True
        )
        coeff = np.array([coefficient for _, coefficient in terms_iterator])

    energy_test = np.sum(coeff * expecval.values)
    energy_ref = np.real(interactionrdm.expectation(qubitoperator))
    assert math.isclose(energy_test, energy_ref)


@pytest.mark.parametrize(
    "interactionrdm, qubitoperator_list, sort_terms",
    [
        (
            rdms,
            h2_hamiltonian_grouped,
            False,
        ),
        (
            rdms,
            h2_hamiltonian_grouped,
            True,
        ),
        (
            rdms,
            QubitOperator("2[]"),
            False,
        ),
    ],
)
def test_get_expectation_values_from_rdms_for_qubitoperator_list(
    interactionrdm, qubitoperator_list, sort_terms
):
    expecval = get_expectation_values_from_rdms_for_qubitoperator_list(
        interactionrdm, qubitoperator_list, sort_terms
    )
    assert len(expecval.values) == np.sum(len(x.terms) for x in qubitoperator_list)
    coeffs = []
    for qubitoperator in qubitoperator_list:
        if not sort_terms:
            coeffs += list(qubitoperator.terms.values())
        else:
            terms_iterator = sorted(
                qubitoperator.terms.items(), key=lambda x: abs(x[1]), reverse=True
            )
            coeff = [coefficient for _, coefficient in terms_iterator]
            coeffs += coeff
    coeffs = np.array(coeffs)

    energy_test = np.sum(coeffs * expecval.values)
    energy_ref = np.sum(
        [np.real(interactionrdm.expectation(x)) for x in qubitoperator_list]
    )
    assert math.isclose(energy_test, energy_ref)


@pytest.mark.parametrize(
    "groups, expecval, variances",
    [
        (
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            None,
            np.array([2.0, 2.0]),
        ),
        (
            [
                QubitOperator("[Z0 Z1] + [Z0] + []"),
                QubitOperator("[X0 X1] + [X0] + 5 []"),
            ],
            None,
            np.array([2.0, 2.0]),
        ),
        (
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.zeros(4)),
            np.array([2.0, 2.0]),
        ),
        (
            [QubitOperator("2 [Z0 Z1] + 3 [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.zeros(4)),
            np.array([13.0, 2.0]),
        ),
        (
            [QubitOperator("2[]")],
            ExpectationValues(np.array([1])),
            np.array([0.0]),
        ),
        (
            [
                QubitOperator("2 [Z0 Z1] + 3 [Z0] + 8[]"),
                QubitOperator("[X0 X1] + [X0] + []"),
            ],
            ExpectationValues(np.asarray([0.0, 0.0, 1.0, 0, 0, 1.0])),
            np.array([13.0, 2.0]),
        ),
    ],
)
def test_compute_group_variances_with_ref(groups, expecval, variances):
    test_variances = compute_group_variances(groups, expecval)
    np.testing.assert_allclose(test_variances, variances)


@pytest.mark.parametrize(
    "groups, expecval, variances",
    [
        (
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.array([-1.5, 0, 0, 0])),
            np.array([2.0, 2.0]),
        ),
        (
            [QubitOperator("2 [Z0 Z1] + 3 [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.array([1.5, 0, 0, 0])),
            np.array([13.0, 2.0]),
        ),
        (
            [QubitOperator("2 [Z0 Z1] + 3 [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.array([1.5, 0, 0, 0])),
            np.array([13.0, 2.0]),
        ),
    ],
)
def test_compute_group_variances_fails_for_invalid_refs(groups, expecval, variances):
    with pytest.raises(ValueError):
        _ = compute_group_variances(groups, expecval)


@pytest.mark.parametrize(
    "groups, expecval",
    [
        (
            group_comeasureable_terms_greedy(h2_hamiltonian, False),
            ExpectationValues(np.ones(15)),
        ),
        (
            group_comeasureable_terms_greedy(h2_hamiltonian, False),
            ExpectationValues(np.zeros(15)),
        ),
        (
            group_comeasureable_terms_greedy(h2_hamiltonian, False),
            ExpectationValues(np.repeat(0.5, 15)),
        ),
        (
            group_comeasureable_terms_greedy(h2_hamiltonian, False),
            get_expectation_values_from_rdms(rdms, h2_hamiltonian, False),
        ),
        (
            group_comeasureable_terms_greedy(h2_hamiltonian, True),
            get_expectation_values_from_rdms(rdms, h2_hamiltonian, True),
        ),
        (
            group_comeasureable_terms_greedy(QubitOperator("2[]"), True),
            get_expectation_values_from_rdms(rdms, QubitOperator("2[]"), True),
        ),
    ],
)
def test_compute_group_variances_without_ref(groups, expecval):
    test_variances = compute_group_variances(groups, expecval)
    test_ham_variance = np.sum(test_variances)
    # Assemble H and compute its variances independently
    ham = QubitOperator()
    for g in groups:
        ham += g
    ham_coeff = np.array(list(ham.terms.values()))
    pauli_var = 1.0 - expecval.values ** 2
    ref_ham_variance = np.sum(ham_coeff ** 2 * pauli_var)
    assert math.isclose(
        test_ham_variance, ref_ham_variance
    )  # this is true as long as the groups do not overlap


@pytest.mark.parametrize(
    "frame_operators, expecval, expected_result",
    [
        (
            h2_hamiltonian_grouped,
            None,
            (
                0.5646124437984263,
                15,
                np.array(
                    [0, 0.03362557, 0.03362557, 0.03362557, 0.03362557, 0.43011016]
                ),
            ),
        ),
        (
            h2_hamiltonian_grouped,
            get_expectation_values_from_rdms(rdms, h2_hamiltonian, False),
            (
                0.06951544260278607,
                15,
                np.array(
                    [0, 0.01154017, 0.01154017, 0.01154017, 0.01154017, 0.02335476]
                ),
            ),
        ),
    ],
)
def test_estimate_nmeas_with_groups(frame_operators, expecval, expected_result):
    K2_ref, nterms_ref, frame_meas_ref = expected_result
    K2, nterms, frame_meas = estimate_nmeas_for_frames(frame_operators, expecval)
    np.testing.assert_allclose(frame_meas, frame_meas_ref)
    assert math.isclose(K2_ref, K2)
    assert nterms_ref == nterms


def test_reorder_fermionic_modes():
    ref_op = get_interaction_operator(
        FermionOperator(
            """
    0.0 [] +
    1.0 [0^ 0] +
    1.0 [1^ 1] +
    1.0 [0^ 1^ 2 3] +
    1.0 [1^ 1^ 2 2]
    """
        )
    )
    reordered_op = get_interaction_operator(
        FermionOperator(
            """
    0.0 [] +
    1.0 [0^ 0] +
    1.0 [2^ 2] +
    1.0 [0^ 2^ 1 3] +
    1.0 [2^ 2^ 1 1]
    """
        )
    )
    spin_block_op = reorder_fermionic_modes(ref_op, [0, 2, 1, 3])
    assert reordered_op == spin_block_op
    spin_block_qubit_op = jordan_wigner(spin_block_op)
    interleaved_qubit_op = jordan_wigner(ref_op)
    spin_block_spectrum = eigenspectrum(spin_block_qubit_op)
    interleaved_spectrum = eigenspectrum(interleaved_qubit_op)
    np.testing.assert_allclose(spin_block_spectrum, interleaved_spectrum)
