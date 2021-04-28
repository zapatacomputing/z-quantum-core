import json
import math
from io import StringIO
from itertools import product
from unittest import mock

import numpy as np
import pytest
from zquantum.core.bitstring_distribution._bitstring_distribution import (
    BitstringDistribution,
    are_keys_binary_strings,
    create_bitstring_distribution_from_probability_distribution,
    evaluate_distribution_distance,
    is_bitstring_distribution,
    is_key_length_fixed,
    is_non_negative,
    is_normalized,
    load_bitstring_distribution,
    load_bitstring_distribution_set,
    normalize_bitstring_distribution,
    save_bitstring_distribution,
    save_bitstring_distribution_set,
)
from zquantum.core.utils import SCHEMA_VERSION


def test_dicts_with_nonnegative_values_are_nonnegative():
    """The is_non_negative function returns True for dicts with nonnegative values."""
    n_elements = 10
    nonnegative_dict = {i: i + 1 for i in range(n_elements)}
    assert is_non_negative(nonnegative_dict)


@pytest.mark.parametrize(
    "dictionary", [{i: -i for i in range(10)}, {0: -1, 1: 2, 3: 0}]
)
def test_dicts_with_some_negative_values_are_not_nonnegative(dictionary):
    """The is_non_negative function returns False for dicts with some negative values."""
    assert not is_non_negative(dictionary)


def test_if_all_keys_have_the_same_length_the_key_length_is_fixed():
    """The is_key_length_fixed returns True if all keys have the same length."""
    assert is_key_length_fixed({"abc": 3, "100": 2, "www": 1})


def test_if_some_keys_have_different_length_the_key_length_is_not_fixed():
    """The is_key_length_fixed returns False if some keys have different length."""
    assert not is_key_length_fixed({"a": 3, "10": 2, "www": 1})


def test_if_dict_keys_have_only_01_characters_the_keys_are_binary_strings():
    """The are_keys_binary_strings returns True for binary string keyed dicts."""
    assert are_keys_binary_strings({"100001": 3, "10": 2, "0101": 1})


def test_if_dict_keys_have_characters_other_than_01_the_keys_are_not_binary_strings():
    """The are_keys_binary_strings returns False for non-binary string keyed dicts."""
    assert not are_keys_binary_strings({"abc": 3, "100": 2, "www": 1})


def test_dict_with_varying_key_length_is_not_bitstring_distributions():
    """Dictionaries with varying key length are not bitstring distributions."""
    assert not is_bitstring_distribution({"100001": 3, "10": 2, "0101": 1})


def test_dict_with_non_binary_string_key_is_not_bitstring_distribution():
    """Dictionaries with non 0-1 chars in their keys are not bitstring distributions."""
    assert not is_bitstring_distribution({"abc": 3, "100": 2, "www": 1})


def test_dicts_with_binary_keys_and_fixed_key_length_are_bitstring_distributions():
    """Binary string keyed dictionaries with fixed key length are bitstring distributions."""
    assert is_bitstring_distribution({"100": 3, "110": 2, "010": 1})


@pytest.mark.parametrize(
    "distribution",
    [
        {"000": 0.1, "111": 0.9},
        {"010": 0.3, "000": 0.2, "111": 0.5},
        {"010": 0.3, "000": 0.2, "111": 0.1, "100": 0.4},
    ],
)
def test_distributions_with_probabilities_summing_to_one_are_normalized(distribution):
    """Distributions with probabilities summing to one are normalized."""
    assert is_normalized(distribution)


@pytest.mark.parametrize(
    "distribution",
    [
        {"000": 0.1, "111": 9},
        {"000": 2, "111": 0.9},
        {"000": 1e-3, "111": 0, "100": 100},
    ],
)
def test_distributions_with_probabilities_not_summing_to_one_are_not_normalized(
    distribution,
):
    """Distributions with probabilities not summing to one are normalized."""
    assert not is_normalized(distribution)


@pytest.mark.parametrize(
    "distribution",
    [
        {"000": 0.1, "111": 9},
        {"000": 2, "111": 0.9},
        {"000": 1e-3, "111": 0, "100": 100},
    ],
)
def test_normalizing_distribution_gives_normalized_distribution(distribution):
    """Normalizing bitstring distribution returns normalized bitstring distribution."""
    assert not is_normalized(distribution)
    normalize_bitstring_distribution(distribution)
    assert is_normalized(distribution)


@pytest.mark.parametrize(
    "prob_dist,expected_bitstring_dist",
    [
        (
            np.asarray([0.25, 0, 0.5, 0.25]),
            BitstringDistribution({"00": 0.25, "01": 0.5, "10": 0.0, "11": 0.25}),
        ),
        (
            np.ones(2 ** 5) / 2 ** 5,
            BitstringDistribution(
                {"".join(string): 1 / 2 ** 5 for string in product("01", repeat=5)}
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
    target_distribution = BitstringDistribution({"0": 10, "1": 5})
    measured_distribution = BitstringDistribution({"0": 10, "1": 5})
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
    """Saving bitstring distribution opens file for writing using context manager."""
    distribution = BitstringDistribution({"000": 0.1, "111": 0.9})
    save_bitstring_distribution(distribution, "/some/path/to/distribution.json")

    mock_open.assert_called_once_with("/some/path/to/distribution.json", "w")
    mock_open().__enter__.assert_called_once()
    mock_open().__exit__.assert_called_once()


def test_saving_bitstring_distribution_set_opens_file_for_writing_using_context_manager(
    mock_open,
):
    """Saving bitstring distribution set opens file for writing using context manager."""
    distributions = [
        BitstringDistribution({"000": 0.1, "111": 0.9}),
        BitstringDistribution({"01000": 0.5, "10110": 0.5}),
    ]
    save_bitstring_distribution_set(
        distributions, "/some/path/to/distribution/set.json"
    )

    mock_open.assert_called_once_with("/some/path/to/distribution/set.json", "w")
    mock_open().__enter__.assert_called_once()
    mock_open().__exit__.assert_called_once()


def test_saving_bitstring_distribution_writes_correct_json_data_to_file(mock_open):
    """Saving bitstring distribution writes correct json dictionary to file."""
    distribution = BitstringDistribution({"000": 0.1, "111": 0.9})

    expected_dict = {
        "bitstring_distribution": distribution.distribution_dict,
        "schema": SCHEMA_VERSION + "-bitstring-probability-distribution",
    }

    save_bitstring_distribution(distribution, "/some/path/to/distribution.json")

    written_data = mock_open().__enter__().write.call_args[0][0]
    assert json.loads(written_data) == expected_dict


def test_saving_bitstring_distribution_set_writes_correct_json_data_to_file(mock_open):
    """Saving bitstring distribution set writes correct list of json dictionaries to file."""
    distributions = [
        BitstringDistribution({"000": 0.1, "111": 0.9}),
        BitstringDistribution({"01000": 0.5, "10110": 0.5}),
    ]

    expected_dict = {
        "bitstring_distribution": [
            distribution.distribution_dict for distribution in distributions
        ],
        "schema": SCHEMA_VERSION + "-bitstring-probability-distribution-set",
    }

    save_bitstring_distribution_set(
        distributions, "/some/path/to/distribution/set.json"
    )

    written_data = mock_open().__enter__().write.call_args[0][0]
    assert json.loads(written_data) == expected_dict


def test_saved_bitstring_distribution_can_be_loaded(mock_open):
    """Saved bitstring distribution can be loaded to obtain the same distribution."""
    fake_file = StringIO()
    mock_open().__enter__.return_value = fake_file
    dist = BitstringDistribution({"000": 0.1, "111": 0.9})

    save_bitstring_distribution(dist, "distribution.json")
    fake_file.seek(0)

    loaded_dist = load_bitstring_distribution(fake_file)
    assert all(
        math.isclose(dist.distribution_dict[key], loaded_dist.distribution_dict[key])
        for key in dist.distribution_dict.keys()
    )

    assert dist.distribution_dict.keys() == loaded_dist.distribution_dict.keys()


def test_saved_bitstring_distribution_set_can_be_loaded(mock_open):
    """Saved bitstring distribution set can be loaded to obtain the same distribution set."""
    fake_file = StringIO()
    mock_open().__enter__.return_value = fake_file
    distributions = [
        BitstringDistribution({"000": 0.1, "111": 0.9}),
        BitstringDistribution({"01000": 0.5, "10110": 0.5}),
    ]

    save_bitstring_distribution_set(distributions, "distributions.json")
    fake_file.seek(0)

    loaded_distributions = load_bitstring_distribution_set(fake_file)
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
        {"000": 0.1, "111": 0.9},
        {"010": 0.3, "111": 0.9},
        {"000": 2, "111": 0.9},
        {"000": 2, "111": 4.9},
        {"000": 0.2, "111": 9},
        {"000": 1e-3, "111": 0},
    ],
)
def test_bitstring_distribution_gets_normalized_by_default(distribution):
    """Constructing bitstring distribution normalizes it by default."""
    distribution = BitstringDistribution(distribution)
    assert is_normalized(distribution.distribution_dict)


def test_bitstring_distribution_keeps_original_dict_if_normalization_should_not_be_performed():
    """Bistring distribution keeps original dict if normalization is not requested."""
    distribution_dict = {"000": 0.1, "111": 9}
    distribution = BitstringDistribution({"000": 0.1, "111": 9}, normalize=False)
    assert distribution.distribution_dict == distribution_dict


@pytest.mark.parametrize(
    "distribution,num_qubits",
    [
        (BitstringDistribution({"00": 0.1, "11": 0.9}), 2),
        (BitstringDistribution({"000": 0.2, "111": 0.8}), 3),
        (BitstringDistribution({"0000": 1e-3, "1111": 0}), 4),
    ],
)
def test_number_of_qubits_in_bitstring_distribution_equals_length_of_keys(
    distribution, num_qubits
):
    """Number of qubits of bitstring distribution is equal to length of keys of distribution."""
    assert distribution.get_qubits_number() == num_qubits
