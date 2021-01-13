import unittest
import os
import random
import numpy as np
import pytest
from scipy.stats import unitary_group
import sympy
import json

from .utils import (
    convert_array_to_dict,
    convert_dict_to_array,
    dec2bin,
    bin2dec,
    is_identity,
    is_unitary,
    compare_unitary,
    RNDSEED,
    ValueEstimate,
    save_value_estimate,
    load_value_estimate,
    save_list,
    load_list,
    create_object,
    get_func_from_specs,
    load_noise_model,
    save_noise_model,
    create_symbols_map,
    save_timing,
    save_nmeas_estimate,
    load_nmeas_estimate,
    SCHEMA_VERSION,
)


class TestUtils(unittest.TestCase):
    def test_real_array_conversion(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        dictionary = convert_array_to_dict(arr)
        new_arr = convert_dict_to_array(dictionary)
        self.assertTrue(np.allclose(arr, new_arr))

    def test_complex_array_conversion(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0j]])
        dictionary = convert_array_to_dict(arr)
        new_arr = convert_dict_to_array(dictionary)
        self.assertTrue(np.allclose(arr, new_arr))

    def test_dec_bin_conversion(self):
        integer = random.randint(1, 10 ** 9)
        integer2 = bin2dec(dec2bin(integer, 30))
        self.assertEqual(integer, integer2)

    def test_is_identity(self):
        # Given
        matrix1 = np.eye(4)
        random.seed(RNDSEED)
        matrix2 = np.array(
            [[random.uniform(-1, 1) for x in range(0, 4)] for y in range(0, 4)]
        )
        # When/Then
        self.assertTrue(is_identity(matrix1))
        self.assertFalse(is_identity(matrix2))

    def test_is_unitary(self):
        # Given
        U = unitary_group.rvs(4)
        # When/Then
        self.assertTrue(is_unitary(U))

    def test_compare_unitary(self):
        # Given
        U1 = unitary_group.rvs(4)
        U2 = unitary_group.rvs(4)
        # When/Then
        self.assertFalse(compare_unitary(U1, U2))

    def test_sample_from_probability_distribution(self):
        pass

    def test_convert_bitstrings_to_tuples(self):
        pass

    def test_convert_tuples_to_bitstrings(self):
        pass

    def test_value_estimate_dict_conversion(self):
        pass

    def test_value_estimate_io(self):
        # Given
        value = -1.0
        precision = 0.1
        value_estimate_object = ValueEstimate(value, precision)

        # When
        save_value_estimate(value_estimate_object, "value_estimate.json")
        value_estimate_object_loaded = load_value_estimate("value_estimate.json")

        # Then
        self.assertEqual(
            value_estimate_object.value, value_estimate_object_loaded.value
        )
        self.assertEqual(
            value_estimate_object.precision, value_estimate_object_loaded.precision
        )

        # Given
        value_estimate_object = ValueEstimate(value)
        # When
        save_value_estimate(value_estimate_object, "value_estimate.json")
        value_estimate_object_loaded = load_value_estimate("value_estimate.json")
        # Then
        self.assertEqual(
            value_estimate_object.value, value_estimate_object_loaded.value
        )
        self.assertEqual(
            value_estimate_object.precision, value_estimate_object_loaded.precision
        )

        # Given
        value = np.float64(-1.0)
        precision = np.float64(0.1)
        value_estimate_object = ValueEstimate(value, precision)

        # When
        save_value_estimate(value_estimate_object, "value_estimate.json")
        value_estimate_object_loaded = load_value_estimate("value_estimate.json")

        # Then
        self.assertEqual(
            value_estimate_object.value, value_estimate_object_loaded.value
        )
        self.assertEqual(
            value_estimate_object.precision, value_estimate_object_loaded.precision
        )

        os.remove("value_estimate.json")

    def test_value_estimate_to_string(self):
        value = -1.0
        precision = 0.1
        value_estimate = ValueEstimate(value, precision)
        self.assertEqual(str(value_estimate), f"{value} ± {precision}")

        value_estimate_no_precision = ValueEstimate(value)
        self.assertEqual(str(value_estimate_no_precision), f"{value}")

    def test_list_io(self):
        # Given
        initial_list = [0.1, 0.3, -0.3]
        # When
        save_list(initial_list, "list.json")
        loaded_list = load_list("list.json")
        # Then
        self.assertListEqual(initial_list, loaded_list)
        os.remove("list.json")

    def test_named_list_io(self):
        # Given
        initial_list = [0.1, 0.3, -0.3]
        # When
        save_list(initial_list, "list.json", "number")
        loaded_list = load_list("list.json")
        # Then
        self.assertListEqual(initial_list, loaded_list)
        # And
        # After manually loading json
        if isinstance("list.json", str):
            with open("list.json", "r") as f:
                data = json.load(f)
        else:
            data = json.load("list.json")
        # Check that
        self.assertEqual(data["schema"], SCHEMA_VERSION + "-number-list")
        os.remove("list.json")

    def test_create_object(self):
        # Given
        n_samples = 100
        function_name = "MockQuantumSimulator"
        specs = {
            "module_name": "zquantum.core.interfaces.mock_objects",
            "function_name": function_name,
            "n_samples": n_samples,
        }

        # When
        mock_simulator = create_object(specs)

        # Then
        self.assertEqual(type(mock_simulator).__name__, function_name)
        self.assertEqual(mock_simulator.n_samples, n_samples)

    def test_get_func_from_specs(self):
        # Given
        function_name = "sum_x_squared"
        specs = {
            "module_name": "zquantum.core.interfaces.optimizer_test",
            "function_name": function_name,
        }
        data = np.array([1.0, 2.0])
        target_value = 5.0
        # When
        function = get_func_from_specs(specs)

        # Then
        self.assertEqual(function.__name__, function_name)
        self.assertEqual(function(data), target_value)

    def test_noise_model_io(self):
        # Given
        module_name = "zquantum.core.testing.mocks"
        function_name = "mock_create_noise_model"
        noise_model_data = {"testing": "data"}

        # When
        save_noise_model(
            noise_model_data,
            module_name,
            function_name,
            "noise_model.json",
        )
        noise_model = load_noise_model("noise_model.json")

        # Then
        self.assertEqual(noise_model, None)
        os.remove("noise_model.json")

    def test_create_symbols_map_with_correct_input(self):
        # Given
        symbol_1 = sympy.Symbol("alpha")
        symbol_2 = sympy.Symbol("beta")
        symbols = [symbol_1, symbol_2]
        params = np.array([1, 2])
        target_symbols_map = [(symbol_1, 1), (symbol_2, 2)]

        # When
        symbols_map = create_symbols_map(symbols, params)

        # Then
        self.assertEqual(symbols_map, target_symbols_map)

    def test_create_symbols_map_with_incorrect_input(self):
        # Given
        symbol_1 = sympy.Symbol("alpha")
        symbols = [symbol_1]
        params = np.array([1, 2])

        # When/Then
        with self.assertRaises(ValueError):
            symbols_map = create_symbols_map(symbols, params)

    def test_save_timing(self):
        walltime = 4.2
        save_timing(walltime, "timing.json")
        with open("timing.json") as f:
            timing = json.load(f)
        self.assertEqual(timing["walltime"], walltime)
        self.assertTrue("schema" in timing)
        os.remove("timing.json")

    def test_save_nmeas_estimate(self):
        K_coeff = 0.5646124437984263
        nterms = 14
        frame_meas = np.array(
            [0.03362557, 0.03362557, 0.03362557, 0.03362557, 0.43011016]
        )
        save_nmeas_estimate(
            nmeas=K_coeff,
            nterms=nterms,
            filename="hamiltonian_analysis.json",
            frame_meas=frame_meas,
        )
        K_coeff_, nterms_, frame_meas_ = load_nmeas_estimate(
            "hamiltonian_analysis.json"
        )
        self.assertEqual(K_coeff, K_coeff_)
        self.assertEqual(nterms, nterms_)
        self.assertListEqual(frame_meas.tolist(), frame_meas_.tolist())
        os.remove("hamiltonian_analysis.json")

        os.remove("timing.json")


def test_arithmetic_on_value_estimate_and_float_gives_the_same_result_as_arithmetic_on_two_floats():
    value = 5.1
    estimate = ValueEstimate(value, precision=None)
    other = 3.4

    assert estimate + other == value + other
    assert estimate - other == value - other
    assert estimate * other == value * other
    assert estimate / other == value / other


def test_value_estimate_with_no_precision_is_equivalent_to_its_raw_value():
    value = 6.193
    estimate = ValueEstimate(value)

    # Note that it is not that obvious that this comparison is symmetric, since we override
    # the __eq__ method in ValueEstimate. The same goes about __ne__ method in the next test.
    assert value == estimate
    assert estimate == value


def test_value_estimate_with_specified_precision_is_not_equal_to_its_raw_value():
    value = 6.193
    estimate = ValueEstimate(value, precision=4)

    assert value != estimate
    assert estimate != value


@pytest.mark.parametrize(
    "estimate_1,estimate_2,expected_result",
    [
        (ValueEstimate(14.1), ValueEstimate(14.1), True),
        (ValueEstimate(12.3, 3), ValueEstimate(12.3, 3), True),
        (ValueEstimate(14.1, 5), ValueEstimate(14.1, 4), False),
        (ValueEstimate(2.5, 3), ValueEstimate(2.5), False),
        (ValueEstimate(0.15, 3), ValueEstimate(1.1, 3), False),
    ],
)
def test_two_value_estimates_are_equal_iff_their_values_and_precisions_are_equal(
    estimate_1, estimate_2, expected_result
):
    assert (estimate_1 == estimate_2) == expected_result


@pytest.mark.parametrize(
    "estimate", [ValueEstimate(2.0), ValueEstimate(5.0, precision=1e-5)]
)
@pytest.mark.parametrize("other_obj", ["test-string", {"foo": 5, "bar": 10}, [1, 2, 3]])
def test_value_estimate_is_not_equivalent_to_an_object_of_non_numeric_type(
    estimate, other_obj
):
    assert estimate != other_obj
