import json
import math
from io import StringIO
from itertools import product
from sys import float_info
from unittest import mock

import numpy as np
import pytest
from zquantum.core.bitstring_distribution import (
    BitstringDistribution,
    change_tuple_dict_keys_to_comma_separated_integers,
    create_bitstring_distribution_from_probability_distribution,
    evaluate_distribution_distance,
    is_measurement_outcome_distribution,
    is_normalized,
    load_measurement_outcome_distribution,
    load_measurement_outcome_distributions,
    normalize_measurement_outcome_distribution,
    preprocess_distibution_dict,
    save_measurement_outcome_distribution,
    save_measurement_outcome_distributions,
)
from zquantum.core.distribution._measurement_outcome_distribution import (
    _are_keys_non_negative_integer_tuples,
    _is_key_length_fixed,
    _is_non_negative,
)
from zquantum.core.utils import SCHEMA_VERSION


class TestInitializations:
    @pytest.mark.parametrize(
        "prob_dist,expected_dist",
        [
            (
                np.asarray([0.25, 0, 0.5, 0.25]),
                BitstringDistribution({"00": 0.25, "10": 0.5, "01": 0.0, "11": 0.25}),
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
        self, prob_dist, expected_dist
    ):
        """Probability distributions is converted to matching bitstring distributions.

        Bitstring distributions constructed from probability distribution should have:
        - keys equal to binary representation of consecutive natural numbers,
        - values corresponding to original probabilities.
        """
        bitstring_dist = create_bitstring_distribution_from_probability_distribution(
            prob_dist
        )
        assert bitstring_dist.distribution_dict == expected_dist.distribution_dict
        assert (
            bitstring_dist.get_number_of_subsystems()
            == expected_dist.get_number_of_subsystems()
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
    def test_distribution_gets_normalized_by_default(self, distribution):
        distribution = BitstringDistribution(distribution)
        assert is_normalized(distribution.distribution_dict)

    def test_original_dict_is_kept_if_normalization_isnt_requested_and_warns(
        self,
    ):
        distribution_dict = {(0, 0, 0): 0.1, (1, 1, 1): 9}
        with pytest.warns(UserWarning):
            distribution = BitstringDistribution(
                {(0, 0, 0): 0.1, (1, 1, 1): 9}, normalize=False
            )
        assert distribution.distribution_dict == distribution_dict

    @pytest.mark.parametrize(
        "distribution,num_qubits",
        [
            (BitstringDistribution({"00": 0.1, "11": 0.9}), 2),
            (BitstringDistribution({"000": 0.2, "111": 0.8}), 3),
            (BitstringDistribution({"0000": 1e-3, "1111": 0}), 4),
        ],
    )
    def test_number_of_qubits_in_distribution_equals_length_of_keys(
        self, distribution, num_qubits
    ):
        assert distribution.get_number_of_subsystems() == num_qubits

    def test_constructor_invalid_distribution_throws_error(self):
        with pytest.raises(RuntimeError):
            BitstringDistribution({"010": 0.1, "1": 0.9})


def test_repr_function_returns_expected_string():
    dictionary = {(0,): 0.1, (1,): 0.9}
    dist = BitstringDistribution(dictionary)

    assert dist.__repr__() == f"BitstringDistribution(input={dictionary})"


class TestNormalization:
    def test_normalizing_normalized_dict_does_nothing(self):
        assert normalize_measurement_outcome_distribution({"0": 1.0}) == {"0": 1.0}

    @pytest.mark.parametrize(
        "dist",
        [
            {"000": 0.1, "111": 9},
            {"000": 2, "111": 0.9},
            {"000": 1e-3, "111": 0, "100": 100},
        ],
    )
    def test_normalizing_distribution_gives_normalized_distribution(self, dist):
        assert not is_normalized(dist)
        normalize_measurement_outcome_distribution(dist)
        assert is_normalized(dist)

    @pytest.mark.parametrize(
        "dist,expected_error_content",
        [
            ({"000": 0.0, "111": 0.0}, "all zero values"),
            ({"000": float_info.min / 2}, "too small values"),
        ],
    )
    def test_normalizing_distribution_raises_error_for_values_with_invalid_norm(
        self, dist, expected_error_content
    ):
        with pytest.raises(ValueError) as error:
            normalize_measurement_outcome_distribution(dist)

        assert expected_error_content in error.value.args[0]


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


class TestSavingDistributions:
    @pytest.fixture
    def mock_open(self):
        mock_open = mock.mock_open()
        with mock.patch(
            "zquantum.core.distribution._measurement_outcome_distribution.open",
            mock_open,
        ):
            yield mock_open

    def test_saving_distribution_opens_file_for_writing_using_context_manager(
        self,
        mock_open,
    ):
        distribution = BitstringDistribution({"000": 0.1, "111": 0.9})
        save_measurement_outcome_distribution(
            distribution, "/some/path/to/distribution.json"
        )

        mock_open.assert_called_once_with("/some/path/to/distribution.json", "w")
        mock_open().__enter__.assert_called_once()
        mock_open().__exit__.assert_called_once()

    def test_saving_distributions_opens_file_for_writing_using_context_manager(
        self,
        mock_open,
    ):
        distributions = [
            BitstringDistribution({"000": 0.1, "111": 0.9}),
            BitstringDistribution({"01000": 0.5, "10110": 0.5}),
        ]
        save_measurement_outcome_distributions(
            distributions, "/some/path/to/distribution/set.json"
        )

        mock_open.assert_called_once_with("/some/path/to/distribution/set.json", "w")
        mock_open().__enter__.assert_called_once()
        mock_open().__exit__.assert_called_once()

    def test_saving_distribution_writes_correct_json_data_to_file(self, mock_open):
        """Saving distribution writes correct json dictionary to file."""
        distribution = BitstringDistribution({"000": 0.1, "111": 0.9})

        preprocessed_dict = change_tuple_dict_keys_to_comma_separated_integers(
            distribution.distribution_dict
        )

        expected_dict = {
            "measurement_outcome_distribution": preprocessed_dict,
            "schema": SCHEMA_VERSION + "-measurement-outcome-probability-distribution",
        }

        save_measurement_outcome_distribution(
            distribution, "/some/path/to/distribution.json"
        )

        written_data = mock_open().__enter__().write.call_args[0][0]
        assert json.loads(written_data) == expected_dict

    def test_saving_distributions_writes_correct_json_data_to_file(self, mock_open):
        distributions = [
            BitstringDistribution({"000": 0.1, "111": 0.9}),
            BitstringDistribution({"01000": 0.5, "10110": 0.5}),
        ]

        expected_dict = {
            "measurement_outcome_distribution": [
                change_tuple_dict_keys_to_comma_separated_integers(
                    distribution.distribution_dict
                )
                for distribution in distributions
            ],
            "schema": SCHEMA_VERSION
            + "-measurement-outcome-probability-distribution-set",
        }

        save_measurement_outcome_distributions(
            distributions, "/some/path/to/distribution/set.json"
        )

        written_data = mock_open().__enter__().write.call_args[0][0]
        assert json.loads(written_data) == expected_dict

    def test_saved_distribution_can_be_loaded_back(self, mock_open):
        fake_file = StringIO()
        mock_open().__enter__.return_value = fake_file
        dist = BitstringDistribution({"000": 0.1, "111": 0.9})

        save_measurement_outcome_distribution(dist, "distribution.json")
        fake_file.seek(0)

        loaded_dist = load_measurement_outcome_distribution(fake_file)
        assert all(
            math.isclose(
                dist.distribution_dict[key], loaded_dist.distribution_dict[key]
            )
            for key in dist.distribution_dict.keys()
        )

        assert dist.distribution_dict.keys() == loaded_dist.distribution_dict.keys()

    def test_saved_distributions_can_be_loaded(self, mock_open):
        fake_file = StringIO()
        mock_open().__enter__.return_value = fake_file
        distributions = [
            BitstringDistribution({"000": 0.1, "111": 0.9}),
            BitstringDistribution({"01000": 0.5, "10110": 0.5}),
        ]

        save_measurement_outcome_distributions(distributions, "distributions.json")
        fake_file.seek(0)

        loaded_distributions = load_measurement_outcome_distributions(fake_file)
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
