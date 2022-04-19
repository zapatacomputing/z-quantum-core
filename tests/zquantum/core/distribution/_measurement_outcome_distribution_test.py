################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import json
import math
from io import StringIO
from itertools import product
from sys import float_info
from unittest import mock

import numpy as np
import pytest
from zquantum.core.distribution._measurement_outcome_distribution import (
    MeasurementOutcomeDistribution,
    _are_keys_non_negative_integer_tuples,
    _is_key_length_fixed,
    _is_non_negative,
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


class TestVerifiersAndValidators:
    @pytest.mark.parametrize(
        "validator,positive_case",
        [
            (_is_non_negative, {i: i + 1 for i in range(10)}),
            (
                _is_key_length_fixed,
                {("a", "b", "c"): 3, (1, 0, 0): 2, ("w", "w", "w"): 1},
            ),
            (
                _are_keys_non_negative_integer_tuples,
                {(1, 0, 0, 0, 0, 1): 3, (1, 99): 2, (0, 45, 36, 1): 1},
            ),
            (
                is_measurement_outcome_distribution,
                {(1, 0, 0): 3, (1, 1, 0): 2, (0, 1, 0): 1},
            ),
            (
                is_measurement_outcome_distribution,
                preprocess_distibution_dict(
                    {
                        "110": 0.5,
                        (1, 0, 0): 0.5,
                    }
                ),
            ),
            (
                is_measurement_outcome_distribution,
                preprocess_distibution_dict(
                    {
                        "1,1,0": 0.5,
                        (1, 0, 0): 0.5,
                    }
                ),
            ),
        ],
    )
    def test_validator_returns_true_for_positive_case(self, validator, positive_case):
        assert validator(positive_case)

    @pytest.mark.parametrize(
        "validator,negative_case",
        [
            (_is_non_negative, {i: -i for i in range(10)}),
            (_is_non_negative, {0: -1, 1: 2, 3: 0}),
            (_is_key_length_fixed, {("a"): 3, (1, 0): 2, ("w", "w", "w"): 1}),
            (
                _are_keys_non_negative_integer_tuples,
                {("a", "b", "c"): 3, (1, 0, 0): 2, ("w", "w", "w"): 1},
            ),
            (
                is_measurement_outcome_distribution,
                {(1, 0, 0, 0, 0, 1): 3, (1, 0): 2, (0, 1, 0, 1): 1},
            ),
            (
                is_measurement_outcome_distribution,
                {("a", "b", "c"): 3, (1, 0, 0): 2, ("w", "w", "w"): 1},
            ),
            (
                is_measurement_outcome_distribution,
                {
                    "abc": 0.5,
                    (1, 0, 0): 0.5,
                },
            ),
            (
                is_measurement_outcome_distribution,
                {
                    "a,b,c": 0.5,
                    (1, 0, 0): 0.5,
                },
            ),
        ],
    )
    def test_validator_returns_false_for_negative_case(self, validator, negative_case):
        assert not validator(negative_case)

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
    def test_distributions_with_probabilities_summing_to_one_are_normalized(
        self,
        distribution,
    ):
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
        self,
        distribution,
    ):
        assert not is_normalized(distribution)

    def test_preprocessor_raises_error_for_invalid_input(self):
        with pytest.raises(RuntimeError):
            preprocess_distibution_dict({1.35: 1.0})


class TestInitializations:
    @pytest.mark.parametrize(
        "prob_dist,expected_dist",
        [
            (
                np.asarray([0.25, 0, 0.5, 0.25]),
                MeasurementOutcomeDistribution(
                    {(0, 0): 0.25, (1, 0): 0.5, (0, 1): 0.0, (1, 1): 0.25}
                ),
            ),
            (
                np.ones(2**5) / 2**5,
                MeasurementOutcomeDistribution(
                    {tup: 1 / 2**5 for tup in product([0, 1], repeat=5)}
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
            {(0, 0, 0): 0.1, (1, 1, 1): 0.9},
            {(0, 1, 0): 0.3, (1, 1, 1): 0.9},
            {(0, 0, 0): 2, (1, 1, 1): 0.9},
            {(0, 0, 0): 2, (1, 1, 1): 4.9},
            {(0, 0, 0): 0.2, (1, 1, 1): 9},
            {(0, 0, 0): 1e-3, (1, 1, 1): 0},
        ],
    )
    def test_distribution_gets_normalized_by_default(self, distribution):
        distribution = MeasurementOutcomeDistribution(distribution)
        assert is_normalized(distribution.distribution_dict)

    def test_original_dict_is_kept_if_normalization_isnt_requested_and_warns(
        self,
    ):
        distribution_dict = {(0, 0, 0): 0.1, (1, 1, 1): 9}
        with pytest.warns(UserWarning):
            distribution = MeasurementOutcomeDistribution(
                {(0, 0, 0): 0.1, (1, 1, 1): 9}, normalize=False
            )
        assert distribution.distribution_dict == distribution_dict

    @pytest.mark.parametrize(
        "distribution,num_qubits",
        [
            (MeasurementOutcomeDistribution({(0, 0): 0.1, (1, 1): 0.9}), 2),
            (MeasurementOutcomeDistribution({(0, 0, 0): 0.2, (1, 1, 1): 0.8}), 3),
            (MeasurementOutcomeDistribution({(0, 0, 0, 0): 1e-3, (1, 1, 1, 1): 0}), 4),
        ],
    )
    def test_number_of_qubits_in_distribution_equals_length_of_keys(
        self, distribution, num_qubits
    ):
        assert distribution.get_number_of_subsystems() == num_qubits

    def test_constructor_invalid_distribution_throws_error(self):
        with pytest.raises(RuntimeError):
            MeasurementOutcomeDistribution({(0, 1, 0): 0.1, (1,): 0.9})


def test_repr_function_returns_expected_string():
    dictionary = {(0,): 0.1, (1,): 0.9}
    dist = MeasurementOutcomeDistribution(dictionary)

    assert dist.__repr__() == f"MeasurementOutcomeDistribution(input={dictionary})"


class TestNormalization:
    def test_normalizing_normalized_dict_does_nothing(self):
        assert normalize_measurement_outcome_distribution({(0,): 1.0}) == {(0,): 1.0}

    @pytest.mark.parametrize(
        "dist",
        [
            {(0, 0, 0): 0.1, (1, 1, 1): 9},
            {(0, 0, 0): 2, (1, 1, 1): 0.9},
            {(0, 0, 0): 1e-3, (1, 1, 1): 0, (1, 0, 0): 100},
        ],
    )
    def test_normalizing_distribution_gives_normalized_distribution(self, dist):
        assert not is_normalized(dist)
        normalize_measurement_outcome_distribution(dist)
        assert is_normalized(dist)

    @pytest.mark.parametrize(
        "dist,expected_error_content",
        [
            ({(0, 0, 0): 0.0, (1, 1, 1): 0.0}, "all zero values"),
            ({(0, 0, 0): float_info.min / 2}, "too small values"),
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
    target_distribution = MeasurementOutcomeDistribution({(0,): 10, (1,): 5})
    measured_distribution = MeasurementOutcomeDistribution({(0,): 10, (1,): 5})
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
        distribution = MeasurementOutcomeDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9})
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
            MeasurementOutcomeDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9}),
            MeasurementOutcomeDistribution(
                {(0, 1, 0, 0, 0): 0.5, (1, 0, 1, 1, 0): 0.5}
            ),
        ]
        save_measurement_outcome_distributions(
            distributions, "/some/path/to/distribution/set.json"
        )

        mock_open.assert_called_once_with("/some/path/to/distribution/set.json", "w")
        mock_open().__enter__.assert_called_once()
        mock_open().__exit__.assert_called_once()

    def test_saving_distribution_writes_correct_json_data_to_file(self, mock_open):
        """Saving distribution writes correct json dictionary to file."""
        distribution = MeasurementOutcomeDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9})

        preprocessed_dict = change_tuple_dict_keys_to_comma_separated_integers(
            distribution.distribution_dict
        )

        expected_dict = {
            "measurement_outcome_distribution": preprocessed_dict,
        }

        save_measurement_outcome_distribution(
            distribution, "/some/path/to/distribution.json"
        )

        written_data = mock_open().__enter__().write.call_args[0][0]
        assert json.loads(written_data) == expected_dict

    def test_saving_distributions_writes_correct_json_data_to_file(self, mock_open):
        distributions = [
            MeasurementOutcomeDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9}),
            MeasurementOutcomeDistribution(
                {(0, 1, 0, 0, 0): 0.5, (1, 0, 1, 1, 0): 0.5}
            ),
        ]

        expected_dict = {
            "measurement_outcome_distribution": [
                change_tuple_dict_keys_to_comma_separated_integers(
                    distribution.distribution_dict
                )
                for distribution in distributions
            ],
        }

        save_measurement_outcome_distributions(
            distributions, "/some/path/to/distribution/set.json"
        )

        written_data = mock_open().__enter__().write.call_args[0][0]
        assert json.loads(written_data) == expected_dict

    def test_saved_distribution_can_be_loaded_back(self, mock_open):
        fake_file = StringIO()
        mock_open().__enter__.return_value = fake_file
        dist = MeasurementOutcomeDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9})

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
            MeasurementOutcomeDistribution({(0, 0, 0): 0.1, (1, 1, 1): 0.9}),
            MeasurementOutcomeDistribution(
                {(0, 1, 0, 0, 0): 0.5, (1, 0, 1, 1, 0): 0.5}
            ),
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
