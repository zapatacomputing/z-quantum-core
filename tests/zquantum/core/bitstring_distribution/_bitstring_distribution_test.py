import json
import math
from io import StringIO
from itertools import product
from unittest import mock

import numpy as np
import pytest
from zquantum.core.bitstring_distribution._bitstring_distribution import (
    BitstringDistribution,
    _change_dict_keys_to_tuple_repr,
    are_keys_binary_strings,
    create_bitstring_distribution_from_probability_distribution,
    evaluate_distribution_distance,
    is_bitstring_distribution,
    is_key_length_fixed,
    is_non_negative,
    is_normalized,
    load_bitstring_distribution,
    load_bitstring_distributions,
    normalize_bitstring_distribution,
    save_bitstring_distribution,
    save_bitstring_distributions,
)
from zquantum.core.utils import SCHEMA_VERSION


def test_dicts_with_nonnegative_values_are_nonnegative():
    n_elements = 10
    nonnegative_dict = {i: i + 1 for i in range(n_elements)}
    assert is_non_negative(nonnegative_dict)


@pytest.mark.parametrize(
    "dictionary", [{i: -i for i in range(10)}, {0: -1, 1: 2, 3: 0}]
)
def test_dicts_with_some_negative_values_are_not_nonnegative(dictionary):
    assert not is_non_negative(dictionary)


def test_if_all_keys_have_the_same_length_the_key_length_is_fixed():
    assert is_key_length_fixed({("a", "b", "c"): 3, (1, 0, 0): 2, ("w", "w", "w"): 1})


def test_if_some_keys_have_different_length_the_key_length_is_not_fixed():
    assert not is_key_length_fixed({("a"): 3, (1, 0): 2, ("w", "w", "w"): 1})


def test_if_dict_keys_have_only_01_characters_the_keys_are_binary_strings():
    assert are_keys_binary_strings({(1, 0, 0, 0, 0, 1): 3, (1, 0): 2, (0, 1, 0, 1): 1})


def test_if_dict_keys_have_characters_other_than_01_the_keys_are_not_binary_strings():
    assert not are_keys_binary_strings(
        {("a", "b", "c"): 3, (1, 0, 0): 2, ("w", "w", "w"): 1}
    )


def test_dict_with_varying_key_length_is_not_bitstring_distributions():
    assert not is_bitstring_distribution(
        {(1, 0, 0, 0, 0, 1): 3, (1, 0): 2, (0, 1, 0, 1): 1}
    )


def test_dict_with_non_binary_string_key_is_not_bitstring_distribution():
    assert not is_bitstring_distribution(
        {("a", "b", "c"): 3, (1, 0, 0): 2, ("w", "w", "w"): 1}
    )


def test_dicts_with_binary_keys_and_fixed_key_length_are_bitstring_distributions():
    assert is_bitstring_distribution({(1, 0, 0): 3, (1, 1, 0): 2, (0, 1, 0): 1})


@pytest.mark.parametrize(
    "distribution",
    [
        {(0, 0, 0): 0.1, (1, 1, 1): 0.9},
        {(0, 1, 0): 0.3, (0, 0, 0): 0.2, (1, 1, 1): 0.5},
        {
            (0, 1, 0): 0.3,
            (0, 0, 0): 0.2,
            (1, 1, 1): 0.1,
            (1, 0, 0): 0.4,
        },
    ],
)
def test_distributions_with_probabilities_summing_to_one_are_normalized(distribution):
    assert is_normalized(distribution)


@pytest.mark.parametrize(
    "distribution",
    [
        {(0, 0, 0): 0.1, (1, 1, 1): 9},
        {(0, 0, 0): 2, (1, 1, 1): 0.9},
        {(0, 0, 0): 1e-3, (1, 1, 1): 0, (1, 0, 0): 100},
    ],
)
def test_distributions_with_probabilities_not_summing_to_one_are_not_normalized(
    distribution,
):
    assert not is_normalized(distribution)


@pytest.mark.parametrize(
    "distribution",
    [
        {(0, 0, 0): 0.1, (1, 1, 1): 9},
        {(0, 0, 0): 2, (1, 1, 1): 0.9},
        {(0, 0, 0): 1e-3, (1, 1, 1): 0, (1, 0, 0): 100},
    ],
)
def test_normalizing_distribution_gives_normalized_distribution(distribution):
    assert not is_normalized(distribution)
    normalize_bitstring_distribution(distribution)
    assert is_normalized(distribution)


@pytest.mark.parametrize(
    "prob_dist,expected_bitstring_dist",
    [
        (
            np.asarray([0.25, 0, 0.5, 0.25]),
            BitstringDistribution(
                {(0, 0): 0.25, (1, 0): 0.5, (0, 1): 0.0, (1, 1): 0.25}
            ),
        ),
        (
            np.ones(2 ** 5) / 2 ** 5,
            BitstringDistribution(
                {tup: 1 / 2 ** 5 for tup in product([0, 1], repeat=5)}
            ),
        ),
    ],
)
def test_constructs_correct_bitstring_distribution_from_probability_distribution(
    prob_dist, expected_bitstring_dist
):
    """Probability distributions is converted to matching bitstring distributions.

    The bitstring distributions constructed from prabability distribution should have:
    - keys equal to binary representation of consecutive natural numbers,
    - values corresponding to original probabilities.
    """
    bitstring_dist = create_bitstring_distribution_from_probability_distribution(
        prob_dist
    )
    assert bitstring_dist.distribution_dict == expected_bitstring_dist.distribution_dict
    assert (
        bitstring_dist.get_qubits_number()
        == expected_bitstring_dist.get_qubits_number()
    )


def test_passed_measure_is_used_for_evaluating_distribution_distance():
    """Evaluating distance distribution uses distance measure passed as an argument."""
    target_distribution = BitstringDistribution({(0,): 10, (1,): 5})
    measured_distribution = BitstringDistribution({(0,): 10, (1,): 5})
    distance_function = mock.Mock()

    distance = evaluate_distribution_distance(
        target_distribution, measured_distribution, distance_function
    )

    distance_function.assert_called_once_with(
        target_distribution, measured_distribution
    )
    assert distance == distance_function.return_value


@pytest.fixture
def mock_open():
    mock_open = mock.mock_open()
    with mock.patch(
        "zquantum.core.bitstring_distribution._bitstring_distribution.open", mock_open
    ):
        yield mock_open


def test_saving_bitstring_distribution_opens_file_for_writing_using_context_manager(
    mock_open,
):
    distribution = BitstringDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9})
    save_bitstring_distribution(distribution, "/some/path/to/distribution.json")

    mock_open.assert_called_once_with("/some/path/to/distribution.json", "w")
    mock_open().__enter__.assert_called_once()
    mock_open().__exit__.assert_called_once()


def test_saving_bitstring_distributions_opens_file_for_writing_using_context_manager(
    mock_open,
):
    distributions = [
        BitstringDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9}),
        BitstringDistribution({(0, 1, 0, 0, 0): 0.5, (1, 0, 1, 1, 0): 0.5}),
    ]
    save_bitstring_distributions(distributions, "/some/path/to/distribution/set.json")

    mock_open.assert_called_once_with("/some/path/to/distribution/set.json", "w")
    mock_open().__enter__.assert_called_once()
    mock_open().__exit__.assert_called_once()


def test_saving_bitstring_distribution_writes_correct_json_data_to_file(mock_open):
    """Saving bitstring distribution writes correct json dictionary to file."""
    distribution = BitstringDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9})

    expected_dict = {
        "bitstring_distribution": _change_dict_keys_to_tuple_repr(
            distribution.distribution_dict
        ),
        "schema": SCHEMA_VERSION + "-bitstring-probability-distribution",
    }

    save_bitstring_distribution(distribution, "/some/path/to/distribution.json")

    written_data = mock_open().__enter__().write.call_args[0][0]
    assert json.loads(written_data) == expected_dict


def test_saving_bitstring_distributions_writes_correct_json_data_to_file(mock_open):
    distributions = [
        BitstringDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9}),
        BitstringDistribution({(0, 1, 0, 0, 0): 0.5, (1, 0, 1, 1, 0): 0.5}),
    ]

    expected_dict = {
        "bitstring_distribution": [
            _change_dict_keys_to_tuple_repr(distribution.distribution_dict)
            for distribution in distributions
        ],
        "schema": SCHEMA_VERSION + "-bitstring-probability-distribution-set",
    }

    save_bitstring_distributions(distributions, "/some/path/to/distribution/set.json")

    written_data = mock_open().__enter__().write.call_args[0][0]
    assert json.loads(written_data) == expected_dict


def test_saved_bitstring_distribution_can_be_loaded_back(mock_open):
    fake_file = StringIO()
    mock_open().__enter__.return_value = fake_file
    dist = BitstringDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9})

    save_bitstring_distribution(dist, "distribution.json")
    fake_file.seek(0)

    loaded_dist = load_bitstring_distribution(fake_file)
    assert all(
        math.isclose(dist.distribution_dict[key], loaded_dist.distribution_dict[key])
        for key in dist.distribution_dict.keys()
    )

    assert dist.distribution_dict.keys() == loaded_dist.distribution_dict.keys()


def test_saved_bitstring_distributions_can_be_loaded(mock_open):
    fake_file = StringIO()
    mock_open().__enter__.return_value = fake_file
    distributions = [
        BitstringDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9}),
        BitstringDistribution({(0, 1, 0, 0, 0): 0.5, (1, 0, 1, 1, 0): 0.5}),
    ]

    save_bitstring_distributions(distributions, "distributions.json")
    fake_file.seek(0)

    loaded_distributions = load_bitstring_distributions(fake_file)
    assert all(
        (
            math.isclose(
                distribution.distribution_dict[key],
                loaded_distribution.distribution_dict[key],
            )
            for key in distribution.distribution_dict.keys()
        )
        for distribution, loaded_distribution in zip(
            distributions, loaded_distributions
        )
    )

    assert all(
        distribution.distribution_dict.keys()
        == loaded_distribution.distribution_dict.keys()
        for distribution, loaded_distribution in zip(
            distributions, loaded_distributions
        )
    )


@pytest.mark.parametrize(
    "distribution",
    [
        {(0, 0, 0): 0.1, (1, 1, 1): 0.9},
        {(0, 1, 0): 0.3, (1, 1, 1): 0.9},
        {(0, 0, 0): 2, (1, 1, 1): 0.9},
        {(0, 0, 0): 2, (1, 1, 1): 4.9},
        {(0, 0, 0): 0.2, (1, 1, 1): 9},
        {(0, 0, 0): 1e-3, (1, 1, 1): 0},
    ],
)
def test_bitstring_distribution_gets_normalized_by_default(distribution):
    distribution = BitstringDistribution(distribution)
    assert is_normalized(distribution.distribution_dict)


def test_bitstring_distribution_keeps_original_dict_if_normalization_isnt_requested():
    distribution_dict = {(0, 0, 0): 0.1, (1, 1, 1): 9}
    distribution = BitstringDistribution(
        {(0, 0, 0): 0.1, (1, 1, 1): 9}, normalize=False
    )
    assert distribution.distribution_dict == distribution_dict


@pytest.mark.parametrize(
    "distribution,num_qubits",
    [
        (BitstringDistribution({(0, 0): 0.1, (1, 1): 0.9}), 2),
        (BitstringDistribution({(0, 0, 0): 0.2, (1, 1, 1): 0.8}), 3),
        (BitstringDistribution({(0, 0, 0, 0): 1e-3, (1, 1, 1, 1): 0}), 4),
    ],
)
def test_number_of_qubits_in_bitstring_distribution_equals_length_of_keys(
    distribution, num_qubits
):
    assert distribution.get_qubits_number() == num_qubits
