import unittest
import os
import numpy as np
import json
import subprocess
from openfermion.ops import QubitOperator, IsingOperator
from .measurement import (
    ExpectationValues,
    Parities,
    save_expectation_values,
    load_expectation_values,
    save_wavefunction,
    load_wavefunction,
    sample_from_wavefunction,
    save_parities,
    load_parities,
    get_parities_from_measurements,
    get_expectation_values_from_parities,
    expectation_values_to_real,
    convert_bitstring_to_int,
    get_expectation_value_from_frequencies,
    Measurements,
    concatenate_expectation_values,
)
from pyquil.wavefunction import Wavefunction

from .bitstring_distribution import BitstringDistribution
from .testing import create_random_wavefunction
from .utils import convert_bitstrings_to_tuples, SCHEMA_VERSION
from collections import Counter


class TestMeasurement(unittest.TestCase):
    def test_expectation_values_io(self):
        expectation_values = np.array([0.0, 0.0, -1.0])
        correlations = []
        correlations.append(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        correlations.append(np.array([[1.0]]))

        covariances = []
        covariances.append(np.array([[0.1, -0.1], [-0.1, 0.1]]))
        covariances.append(np.array([[0.1]]))

        expectation_values_object = ExpectationValues(
            expectation_values, correlations, covariances
        )

        save_expectation_values(expectation_values_object, "expectation_values.json")
        expectation_values_object_loaded = load_expectation_values(
            "expectation_values.json"
        )

        self.assertTrue(
            np.allclose(
                expectation_values_object.values,
                expectation_values_object_loaded.values,
            )
        )
        self.assertEqual(
            len(expectation_values_object.correlations),
            len(expectation_values_object_loaded.correlations),
        )
        self.assertEqual(
            len(expectation_values_object.covariances),
            len(expectation_values_object_loaded.covariances),
        )
        for i in range(len(expectation_values_object.correlations)):
            self.assertTrue(
                np.allclose(
                    expectation_values_object.correlations[i],
                    expectation_values_object_loaded.correlations[i],
                )
            )
        for i in range(len(expectation_values_object.covariances)):
            self.assertTrue(
                np.allclose(
                    expectation_values_object.covariances[i],
                    expectation_values_object_loaded.covariances[i],
                )
            )

        os.remove("expectation_values.json")

    def test_real_wavefunction_io(self):
        wf = Wavefunction([0, 1, 0, 0, 0, 0, 0, 0])
        save_wavefunction(wf, "wavefunction.json")
        loaded_wf = load_wavefunction("wavefunction.json")
        self.assertTrue(np.allclose(wf.amplitudes, loaded_wf.amplitudes))
        os.remove("wavefunction.json")

    def test_imag_wavefunction_io(self):
        wf = Wavefunction([0, 1j, 0, 0, 0, 0, 0, 0])
        save_wavefunction(wf, "wavefunction.json")
        loaded_wf = load_wavefunction("wavefunction.json")
        self.assertTrue(np.allclose(wf.amplitudes, loaded_wf.amplitudes))
        os.remove("wavefunction.json")

    def test_sample_from_wavefunction(self):
        wavefunction = create_random_wavefunction(4)

        samples = sample_from_wavefunction(wavefunction, 100000)
        sampled_dict = Counter(samples)

        sampled_probabilities = []
        for num in range(len(wavefunction) ** 2):
            bitstring = format(num, "b")
            while len(bitstring) < len(wavefunction):
                bitstring = "0" + bitstring
            # NOTE: our indexing places the state of qubit i at the ith index of the tuple. Hence |01> will result in
            # the tuple (1, 0)
            bitstring = bitstring[::-1]
            measurement = convert_bitstrings_to_tuples([bitstring])[0]
            sampled_probabilities.append(sampled_dict[measurement] / 100000)

        probabilities = wavefunction.probabilities()
        for sampled_prob, exact_prob in zip(sampled_probabilities, probabilities):
            self.assertAlmostEqual(sampled_prob, exact_prob, 2)

    def test_sample_from_wavefunction_column_vector(self):
        n_qubits = 4
        # NOTE: our indexing places the state of qubit i at the ith index of the tuple. Hence |01> will result in
        # the tuple (1, 0)
        expected_bitstring = (1, 0, 0, 0)
        amplitudes = np.array([0] * (2 ** n_qubits)).reshape(2 ** n_qubits, 1)
        amplitudes[1] = 1  # |0001> will be measured in all cases.
        wavefunction = Wavefunction(amplitudes)
        sample = set(sample_from_wavefunction(wavefunction, 500))
        self.assertEqual(len(sample), 1)
        self.assertEqual(sample.pop(), expected_bitstring)

    def test_sample_from_wavefunction_row_vector(self):
        n_qubits = 4
        # NOTE: our indexing places the state of qubit i at the ith index of the tuple. Hence |01> will result in
        # the tuple (1, 0)
        expected_bitstring = (1, 0, 0, 0)
        amplitudes = np.array([0] * (2 ** n_qubits))
        amplitudes[1] = 1  # |0001> will be measured in all cases.
        wavefunction = Wavefunction(amplitudes)
        sample = set(sample_from_wavefunction(wavefunction, 500))
        self.assertEqual(len(sample), 1)
        self.assertEqual(sample.pop(), expected_bitstring)

    def test_sample_from_wavefunction_list(self):
        n_qubits = 4
        # NOTE: our indexing places the state of qubit i at the ith index of the tuple. Hence |01> will result in
        # the tuple (1, 0)
        expected_bitstring = (1, 0, 0, 0)
        amplitudes = [0] * (2 ** n_qubits)
        amplitudes[1] = 1  # |0001> will be measured in all cases.
        wavefunction = Wavefunction(amplitudes)
        sample = set(sample_from_wavefunction(wavefunction, 500))
        self.assertEqual(len(sample), 1)
        self.assertEqual(sample.pop(), expected_bitstring)

    def test_parities_io(self):
        measurements = [(1, 0), (1, 0), (0, 1), (0, 0)]
        op = IsingOperator("[Z0] + [Z1] + [Z0 Z1]")
        parities = get_parities_from_measurements(measurements, op)
        save_parities(parities, "parities.json")
        loaded_parities = load_parities("parities.json")
        self.assertTrue(np.allclose(parities.values, loaded_parities.values))
        self.assertEqual(len(parities.correlations), len(loaded_parities.correlations))
        for i in range(len(parities.correlations)):
            self.assertTrue(
                np.allclose(parities.correlations[i], loaded_parities.correlations[i])
            )
        os.remove("parities.json")

    def test_get_expectation_values_from_parities(self):
        parities = Parities(values=np.array([[18, 50], [120, 113], [75, 26]]))
        expectation_values = get_expectation_values_from_parities(parities)

        self.assertEqual(len(expectation_values.values), 3)
        self.assertAlmostEqual(expectation_values.values[0], -0.47058823529411764)
        self.assertAlmostEqual(expectation_values.values[1], 0.030042918454935622)
        self.assertAlmostEqual(expectation_values.values[2], 0.48514851485148514)

        self.assertEqual(len(expectation_values.covariances), 3)
        self.assertTrue(
            np.allclose(
                expectation_values.covariances[0], np.array([[0.014705882352941176]])
            )
        )
        self.assertTrue(
            np.allclose(expectation_values.covariances[1], np.array([[0.00428797]]))
        )

        self.assertTrue(
            np.allclose(expectation_values.covariances[2], np.array([[0.0075706]]))
        )

    def test_expectation_values_to_real(self):
        # Given
        expectation_values = ExpectationValues(
            np.array([0.0 + 0.1j, 0.0 + 1e-10j, -1.0])
        )
        target_expectation_values = ExpectationValues(np.array([0.0, 0.0, -1.0]))

        # When
        real_expectation_values = expectation_values_to_real(expectation_values)

        # Then
        for value in expectation_values.values:
            self.assertNotIsInstance(value, complex)
        np.testing.assert_array_equal(
            real_expectation_values.values, target_expectation_values.values
        )

    def test_convert_bitstring_to_int(self):
        bitstring = (0, 1, 0, 1, 0, 1)
        self.assertEqual(convert_bitstring_to_int(bitstring), 42)

    def test_get_expectation_value_from_frequencies(self):
        bitstrings = ['001', '001', '110', '000']
        bitstring_frequencies = dict(Counter(bitstrings))
        marked_qubits = (1, 2)
        self.assertAlmostEqual(
            get_expectation_value_from_frequencies(
                marked_qubits, bitstring_frequencies
            ),
            -0.5,
        )

    def test_measurement_class_io(self):
        # Given
        measurements_data = {
            "schema": SCHEMA_VERSION + "-measurements",
            "counts": {
                "000": 1,
                "001": 2,
                "010": 1,
                "011": 1,
                "100": 1,
                "101": 1,
                "110": 1,
                "111": 1,
            },
            "bitstrings": [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
            ],
        }
        input_filename = "measurements_input_test.json"
        output_filename = "measurements_output_test.json"

        with open(input_filename, "w") as f:
            f.write(json.dumps(measurements_data, indent=2))

        # When
        measurements = Measurements.load_from_file(input_filename)
        measurements.save(output_filename)

        # Then
        with open(output_filename, "r") as f:
            output_data = json.load(f)
        self.assertEqual(measurements_data, output_data)

    def test_measurement_class_save_for_numpy_integers(self):
        # Given
        target_bitstrings = [(0, 0, 0)]
        input_bitstrings = [(np.int8(0), np.int8(0), np.int8(0))]

        filename = "measurementstest.json"
        measurements = Measurements(input_bitstrings)
        target_measurements = Measurements(target_bitstrings)

        # When
        measurements.save(filename)

        # Then
        recreated_measurements = Measurements.load_from_file(filename)
        self.assertEqual(
            target_measurements.bitstrings, recreated_measurements.bitstrings
        )
        os.remove("measurementstest.json")

    def test_measurement_class_intialize_with_bitstrings(self):
        # Given
        bitstrings = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]

        # When
        measurements = Measurements(bitstrings=bitstrings)

        # Then
        self.assertEqual(
            measurements.bitstrings,
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
        )

    def test_measurement_class_intialize_with_counts(self):
        # Given
        counts = {
            "000": 1,
            "001": 2,
            "010": 1,
            "011": 1,
            "100": 1,
            "101": 1,
            "110": 1,
            "111": 1,
        }

        # When
        measurements = Measurements.from_counts(counts)

        # Then
        self.assertEqual(
            measurements.bitstrings,
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
        )

    def test_measurement_class_bitstrings(self):
        # Given
        measurements_data = {
            "schema": SCHEMA_VERSION + "-measurements",
            "counts": {
                "000": 1,
                "001": 2,
                "010": 1,
                "011": 1,
                "100": 1,
                "101": 1,
                "110": 1,
                "111": 1,
            },
            "bitstrings": [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
            ],
        }
        input_filename = "measurements_input_test.json"

        with open(input_filename, "w") as f:
            f.write(json.dumps(measurements_data, indent=2))
        measurements = Measurements.load_from_file(input_filename)

        # When/Then
        self.assertEqual(
            measurements.bitstrings,
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
                (1, 0, 1),
                (0, 0, 1),
            ],
        )

    def test_measurement_class_get_counts(self):
        # Given
        measurements_data = {
            "schema": SCHEMA_VERSION + "-measurements",
            "counts": {
                "000": 1,
                "001": 2,
                "010": 1,
                "011": 1,
                "100": 1,
                "101": 1,
                "110": 1,
                "111": 1,
            },
            "bitstrings": [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
            ],
        }
        input_filename = "measurements_input_test.json"

        with open(input_filename, "w") as f:
            f.write(json.dumps(measurements_data, indent=2))
        measurements = Measurements.load_from_file(input_filename)

        # When
        counts = measurements.get_counts()

        # Then
        self.assertEqual(measurements_data["counts"], counts)

    def test_measurement_class_get_distribution(self):
        # Given
        measurements_data = {
            "schema": SCHEMA_VERSION + "-measurements",
            "counts": {
                "000": 1,
                "001": 2,
                "010": 1,
                "011": 1,
                "100": 1,
                "101": 1,
                "110": 1,
                "111": 1,
            },
            "bitstrings": [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
            ],
        }
        input_filename = "measurements_input_test.json"

        with open(input_filename, "w") as f:
            f.write(json.dumps(measurements_data, indent=2))
        measurements = Measurements.load_from_file(input_filename)

        # When
        distribution = measurements.get_distribution()

        # Then
        self.assertEqual(
            distribution.distribution_dict,
            {
                "000": 1 / 9,
                "001": 2 / 9,
                "010": 1 / 9,
                "011": 1 / 9,
                "011": 1 / 9,
                "100": 1 / 9,
                "101": 1 / 9,
                "110": 1 / 9,
                "111": 1 / 9,
            },
        )

    def test_measurement_class_add_counts(self):
        # Given
        measurements = Measurements()
        measurements_counts = {
            "000": 1,
            "001": 2,
            "010": 1,
            "011": 1,
            "100": 1,
            "101": 1,
            "110": 1,
            "111": 1,
        }

        # When
        measurements.add_counts(measurements_counts)

        # Then
        self.assertEqual(
            measurements.bitstrings,
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
        )
        self.assertEqual(
            measurements.get_counts(),
            {
                "000": 1,
                "001": 2,
                "010": 1,
                "011": 1,
                "100": 1,
                "101": 1,
                "110": 1,
                "111": 1,
            },
        )

    def test_measurement_class_add_measurements(self):
        # Given
        measurements = Measurements()
        bitstrings = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]

        # When
        measurements.bitstrings = bitstrings

        # Then
        self.assertEqual(
            measurements.bitstrings,
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
        )

        # When
        measurements.bitstrings += bitstrings

        # Then
        self.assertEqual(
            measurements.bitstrings,
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
        )

    def test_get_expectation_values_from_measurements(self):
        # Given
        measurements = Measurements(
            [(0, 1, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0), (1, 1, 1)]
        )
        ising_operator = IsingOperator("10[] + [Z0 Z1] - 10[Z1 Z2]")
        target_expectation_values = np.array([10, 0.2, -2])
        target_correlations = np.array([[100, 2, -20], [2, 1, -10], [-20, -10, 100]])
        denominator = len(measurements.bitstrings)
        covariance_11 = (
            target_correlations[1, 1] - target_expectation_values[1] ** 2
        ) / denominator
        covariance_12 = (
            target_correlations[1, 2]
            - target_expectation_values[1] * target_expectation_values[2]
        ) / denominator
        covariance_22 = (
            target_correlations[2, 2] - target_expectation_values[2] ** 2
        ) / denominator

        target_covariances = np.array(
            [
                [0, 0, 0],
                [0, covariance_11, covariance_12],
                [0, covariance_12, covariance_22],
            ]
        )

        # When
        expectation_values = measurements.get_expectation_values(ising_operator, False)
        # Then
        np.testing.assert_array_equal(
            expectation_values.values, target_expectation_values
        )
        self.assertEqual(len(expectation_values.correlations), 1)
        np.testing.assert_array_equal(
            expectation_values.correlations[0], target_correlations
        )
        self.assertEqual(len(expectation_values.covariances), 1)
        np.testing.assert_array_equal(
            expectation_values.covariances[0], target_covariances
        )

    def test_get_expectation_values_from_measurements_with_bessel_correction(self):
        # Given
        measurements = Measurements(
            [(0, 1, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0), (1, 1, 1)]
        )
        ising_operator = IsingOperator("10[] + [Z0 Z1] - 10[Z1 Z2]")
        target_expectation_values = np.array([10, 0.2, -2])
        target_correlations = np.array([[100, 2, -20], [2, 1, -10], [-20, -10, 100]])
        denominator = len(measurements.bitstrings) - 1
        covariance_11 = (
            target_correlations[1, 1] - target_expectation_values[1] ** 2
        ) / denominator
        covariance_12 = (
            target_correlations[1, 2]
            - target_expectation_values[1] * target_expectation_values[2]
        ) / denominator
        covariance_22 = (
            target_correlations[2, 2] - target_expectation_values[2] ** 2
        ) / denominator

        target_covariances = np.array(
            [
                [0, 0, 0],
                [0, covariance_11, covariance_12],
                [0, covariance_12, covariance_22],
            ]
        )

        # When
        expectation_values = measurements.get_expectation_values(ising_operator, True)
        # Then
        np.testing.assert_array_equal(
            expectation_values.values, target_expectation_values
        )
        self.assertEqual(len(expectation_values.correlations), 1)
        np.testing.assert_array_equal(
            expectation_values.correlations[0], target_correlations
        )
        self.assertEqual(len(expectation_values.covariances), 1)
        np.testing.assert_array_equal(
            expectation_values.covariances[0], target_covariances
        )

    def test_concatenate_expectation_values(self):
        expectation_values_set = [
            ExpectationValues(np.array([1.0, 2.0])),
            ExpectationValues(np.array([3.0, 4.0])),
        ]

        combined_expectation_values = concatenate_expectation_values(
            expectation_values_set
        )
        self.assertTrue(combined_expectation_values.correlations is None)
        self.assertTrue(combined_expectation_values.covariances is None)
        self.assertTrue(
            np.allclose(combined_expectation_values.values, [1.0, 2.0, 3.0, 4.0])
        )

    def test_concatenate_expectation_values_with_cov_and_corr(self):
        expectation_values_set = [
            ExpectationValues(
                np.array([1.0, 2.0]),
                covariances=[np.array([[0.1, 0.2], [0.3, 0.4]])],
                correlations=[np.array([[-0.1, -0.2], [-0.3, -0.4]])],
            ),
            ExpectationValues(
                np.array([3.0, 4.0]),
                covariances=[np.array([[0.1]]), np.array([[0.2]])],
                correlations=[np.array([[-0.1]]), np.array([[-0.2]])],
            ),
        ]
        combined_expectation_values = concatenate_expectation_values(
            expectation_values_set
        )
        self.assertEqual(len(combined_expectation_values.covariances), 3)
        self.assertTrue(
            np.allclose(
                combined_expectation_values.covariances[0], [[0.1, 0.2], [0.3, 0.4]]
            )
        )
        self.assertTrue(
            np.allclose(combined_expectation_values.covariances[1], [[0.1]])
        )
        self.assertTrue(
            np.allclose(combined_expectation_values.covariances[2], [[0.2]])
        )

        self.assertEqual(len(combined_expectation_values.correlations), 3)
        self.assertTrue(
            np.allclose(
                combined_expectation_values.correlations[0],
                [[-0.1, -0.2], [-0.3, -0.4]],
            )
        )
        self.assertTrue(
            np.allclose(combined_expectation_values.correlations[1], [[-0.1]])
        )
        self.assertTrue(
            np.allclose(combined_expectation_values.correlations[2], [[-0.2]])
        )

        self.assertTrue(
            np.allclose(combined_expectation_values.values, [1.0, 2.0, 3.0, 4.0])
        )

    def tearDown(self):
        subprocess.run(
            ["rm", "measurements_input_test.json", "measurements_output_test.json"]
        )
