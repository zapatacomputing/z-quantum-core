import unittest
import numpy as np
import subprocess

import pytest

from .bitstring_distribution import (
    is_non_negative,
    is_key_length_fixed,
    are_keys_binary_strings,
    is_bitstring_distribution,
    is_normalized,
    normalize_bitstring_distribution,
    save_bitstring_distribution,
    load_bitstring_distribution,
    create_bitstring_distribution_from_probability_distribution,
    compute_clipped_negative_log_likelihood,
    evaluate_distribution_distance,
    BitstringDistribution,
)
from .utils import SCHEMA_VERSION


def test_dicts_with_nonnegative_values_are_correctly_classified():
    """The is_non_negative function should return True for dicts with nonnegative values."""
    n_elements = 10
    nonnegative_dict = {i: i + 1 for i in range(n_elements)}
    assert is_non_negative(nonnegative_dict)


@pytest.mark.parametrize("dictionary", [{i: -i for i in range(10)}, {0: -1, 1: 2, 3: 0}])
def test_dicts_with_some_negative_values_are_correctly_classified(dictionary):
    """The is_non_negative function should return False for dicts with some negative values."""
    assert not is_non_negative(dictionary)


def test_dicts_with_fixed_key_length_are_correctly_classified():
    """The is_key_length_fixed should return True if all keys have the same length."""
    assert is_key_length_fixed({"abc": 3, "100": 2, "www": 1})


def test_dicts_with_variable_key_length_are_correctly_classified():
    """The is_key_length_fixed should return False if some keys have different length."""
    assert not is_key_length_fixed({"a": 3, "10": 2, "www": 1})

class TestBitstringDistributionUtils(unittest.TestCase):

    def test_are_keys_binary_strings(self):
        # Given a dictionary with keys that are binary strings
        dict_a = {"100001": 3, "10": 2, "0101": 1}
        # When calling are_keys_binary_strings
        # Then the return value is true
        self.assertEqual(are_keys_binary_strings(dict_a), True)

        # Given a dictionary with keys that are NOT binary strings
        dict_b = {"abc": 3, "100": 2, "www": 1}
        # When calling are_keys_binary_strings
        # Then the return value is false
        self.assertEqual(are_keys_binary_strings(dict_b), False)

    def test_is_bitstring_distribution(self):
        # Given a dictionary that does NOT represent a bitstring distribution
        dict_a = {"abc": 3, "100": 2, "www": 1}
        dict_b = {"100001": 3, "10": 2, "0101": 1}
        # When calling is_bitstring_distribution
        # Then the return value is false
        self.assertEqual(is_bitstring_distribution(dict_a), False)
        self.assertEqual(is_bitstring_distribution(dict_b), False)

        # Given a dictionary that represents a bitstring distribution
        dict_c = {"100": 3, "110": 2, "010": 1}
        # When calling is_bitstring_distribution
        # Then the return value is true
        self.assertEqual(is_bitstring_distribution(dict_c), True)

    def test_is_normalized(self):
        # Given dictionaries representing normalized bitstring distributions
        l = [
            {"000": 0.1, "111": 0.9},
            {"010": 0.3, "000": 0.2, "111": 0.5},
            {"010": 0.3, "000": 0.2, "111": 0.1, "100": 0.4},
        ]
        for distr in l:
            # When calling is_normalized
            # Then the return value is true
            self.assertEqual(is_normalized(distr), True)

        # Given dictionaries representing bitstring distributions that are not normalized
        exc = [
            {"000": 0.1, "111": 9},
            {"000": 2, "111": 0.9},
            {"000": 1e-3, "111": 0, "100": 100},
        ]
        for distr in exc:
            # When calling is_normalized
            # Then the return value is false
            self.assertEqual(is_normalized(distr), False)

    def test_normalize_bitstring_distribution(self):
        # Given dictionaries representing bitstring distributions that are not normalized
        exc = [
            {"000": 0.1, "111": 9},
            {"000": 2, "111": 0.9},
            {"000": 1e-3, "111": 0, "100": 100},
        ]
        for distr in exc:
            # When calling is_normalized
            # Then the return value is false
            self.assertEqual(is_normalized(distr), False)

            normalize_bitstring_distribution(distr)

            # When calling is_normalized after calling normalize_bitstring_distribution
            # Then the return value is true
            self.assertEqual(is_normalized(distr), True)

    def test_create_bitstring_distribution_from_probability_distribution(self):
        # Given a probability distribution
        prob_distribution = np.asarray([0.25, 0, 0.5, 0.25])
        # When calling create_bitstring_distribution_from_probability_distribution
        bitstring_dist = create_bitstring_distribution_from_probability_distribution(
            prob_distribution
        )

        # Then the returned object is an instance of BitstringDistribution with the correct values
        self.assertEqual(type(bitstring_dist), BitstringDistribution)
        self.assertEqual(bitstring_dist.get_qubits_number(), 2)
        self.assertEqual(bitstring_dist.distribution_dict["00"], 0.25)
        self.assertEqual(bitstring_dist.distribution_dict["01"], 0.5)
        self.assertEqual(bitstring_dist.distribution_dict["11"], 0.25)

    def test_create_bitstring_distribution_from_probability_distribution_5_qubits(self):
        # Given a probability distribution
        prob_distribution = np.ones(5 ** 2) / 5 ** 2
        # When calling create_bitstring_distribution_from_probability_distribution
        bitstring_dist = create_bitstring_distribution_from_probability_distribution(
            prob_distribution
        )

        # Then the returned object is an instance of BitstringDistribution with the correct values
        self.assertEqual(type(bitstring_dist), BitstringDistribution)
        self.assertEqual(bitstring_dist.get_qubits_number(), 5)
        self.assertEqual(bitstring_dist.distribution_dict["00000"], 1 / 5 ** 2)

    def test_compute_clipped_negative_log_likelihood(self):
        # Given a target bitstring distribution and a measured bitstring distribution
        target_distr = BitstringDistribution({"000": 0.5, "111": 0.5})
        measured_distr = BitstringDistribution({"000": 0.1, "111": 0.9})

        # When calling compute_clipped_negative_log_likelihood
        clipped_log_likelihood = compute_clipped_negative_log_likelihood(
            target_distr, measured_distr, epsilon=0.1
        )

        # Then the clipped log likelihood calculated is correct
        self.assertEqual(clipped_log_likelihood, 1.203972804325936)

    def test_evaluate_distribution_distance(self):
        # Given
        target_dict = {"0": 10, "1": 5}
        measured_dict = {"0": 10, "1": 5}
        target_distribution = BitstringDistribution(target_dict)
        measured_distribution = BitstringDistribution(measured_dict)
        target_distance = 0.6365141682948128

        # When
        distance = evaluate_distribution_distance(
            target_distribution,
            measured_distribution,
            compute_clipped_negative_log_likelihood,
        )

        # Then
        self.assertAlmostEqual(distance, target_distance)

        # Testing assertions
        # When/Then
        with self.assertRaises(TypeError):
            evaluate_distribution_distance(
                target_distribution,
                measured_dict,
                compute_clipped_negative_log_likelihood,
            )

        # When/Then
        with self.assertRaises(TypeError):
            evaluate_distribution_distance(
                target_dict,
                measured_distribution,
                compute_clipped_negative_log_likelihood,
            )

        # Given
        measured_dict = {"00": 10, "11": 5}
        measured_distribution = BitstringDistribution(measured_dict)

        # When/Then
        with self.assertRaises(RuntimeError):
            evaluate_distribution_distance(
                target_distribution,
                measured_distribution,
                compute_clipped_negative_log_likelihood,
            )

        # Given
        measured_dict = {"0": 10, "1": 5}
        measured_distribution = BitstringDistribution(measured_dict, normalize=False)

        # When/Then
        with self.assertRaises(RuntimeError):
            evaluate_distribution_distance(
                target_distribution,
                measured_distribution,
                compute_clipped_negative_log_likelihood,
            )


class TestBitstringDistribution(unittest.TestCase):
    def test_bitstring_distribution_io(self):
        # Given a BitstringDistribution object
        distr = BitstringDistribution({"000": 0.1, "111": 0.9})
        # When calling save_bitstring_distribution and then load_bitstring_distribution
        save_bitstring_distribution(distr, "distr_test.json")
        new_distr = load_bitstring_distribution("distr_test.json")

        # Then the resulting two objects have identical key-value pairs
        for bitstring in distr.distribution_dict:
            self.assertAlmostEqual(
                distr.distribution_dict[bitstring],
                new_distr.distribution_dict[bitstring],
            )
        for bitstring in new_distr.distribution_dict:
            self.assertAlmostEqual(
                distr.distribution_dict[bitstring],
                new_distr.distribution_dict[bitstring],
            )

    def test_bitstring_distribution_normalization(self):
        # Given a list of BistringDistibutions that are built from bitstring distribution dictionaries that are both normalized and not normalized
        # When not passing the normalize=False flag to the constructor
        l = [
            BitstringDistribution({"000": 0.1, "111": 0.9}),
            BitstringDistribution({"010": 0.3, "111": 0.9}),
            BitstringDistribution({"000": 2, "111": 0.9}),
            BitstringDistribution({"000": 2, "111": 4.9}),
            BitstringDistribution({"000": 0.2, "111": 9}),
            BitstringDistribution({"000": 1e-3, "111": 0}),
        ]
        for distr in l:
            # Then the distributions are all normalized
            self.assertAlmostEqual(sum(distr.distribution_dict.values()), 1)
            self.assertEqual(is_bitstring_distribution(distr.distribution_dict), True)

        # Given a BistringDistibutions object that is built from a bitstring distribution dictionary that is not normalized
        # When passing the normalize=False flag to the constructor
        exc = BitstringDistribution({"000": 0.1, "111": 9}, normalize=False)
        # Then the distributions are not normalized
        self.assertNotEqual(sum(exc.distribution_dict.values()), 1)
        self.assertEqual(is_bitstring_distribution(exc.distribution_dict), True)

    def test_get_qubits_number(self):
        # Given a list of BistringDistributions with different qubit numbers
        l = [
            BitstringDistribution({"00": 0.1, "11": 0.9}),
            BitstringDistribution({"000": 0.2, "111": 0.8}),
            BitstringDistribution({"0000": 1e-3, "1111": 0}),
        ]
        i = 2
        for distr in l:
            # When calling BitstringDistribution.get_qubits_number
            # Then the returned integer is the number of qubits in the distribution keys
            self.assertEqual(distr.get_qubits_number(), i)
            i += 1

    def tearDown(self):
        subprocess.run(["rm", "distr_test.json"])
        return super().tearDown()
