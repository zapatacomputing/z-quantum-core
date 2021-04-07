import json
import os
import random
from collections import Counter

import numpy as np
import pytest
from openfermion.ops import IsingOperator
from pyquil.wavefunction import Wavefunction
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.measurement import (
    ExpectationValues,
    Measurements,
    Parities,
    check_parity,
    concatenate_expectation_values,
    convert_bitstring_to_int,
    expectation_values_to_real,
    get_expectation_value_from_frequencies,
    get_expectation_values_from_parities,
    get_parities_from_measurements,
    load_expectation_values,
    load_parities,
    load_wavefunction,
    sample_from_wavefunction,
    save_expectation_values,
    save_parities,
    save_wavefunction,
)
from zquantum.core.testing import create_random_wavefunction
from zquantum.core.utils import RNDSEED, SCHEMA_VERSION, convert_bitstrings_to_tuples


def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def test_expectation_values_io():
    expectation_values = np.array([0.0, 0.0, -1.0])
    correlations = []
    correlations.append(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    correlations.append(np.array([[1.0]]))

    estimator_covariances = []
    estimator_covariances.append(np.array([[0.1, -0.1], [-0.1, 0.1]]))
    estimator_covariances.append(np.array([[0.1]]))

    expectation_values_object = ExpectationValues(
        expectation_values, correlations, estimator_covariances
    )

    save_expectation_values(expectation_values_object, "expectation_values.json")
    expectation_values_object_loaded = load_expectation_values(
        "expectation_values.json"
    )

    assert np.allclose(
        expectation_values_object.values,
        expectation_values_object_loaded.values,
    )
    assert len(expectation_values_object.correlations) == len(
        expectation_values_object_loaded.correlations
    )
    assert len(expectation_values_object.estimator_covariances) == len(
        expectation_values_object_loaded.estimator_covariances
    )
    for i in range(len(expectation_values_object.correlations)):
        assert np.allclose(
            expectation_values_object.correlations[i],
            expectation_values_object_loaded.correlations[i],
        )
    for i in range(len(expectation_values_object.estimator_covariances)):
        assert np.allclose(
            expectation_values_object.estimator_covariances[i],
            expectation_values_object_loaded.estimator_covariances[i],
        )

    remove_file_if_exists("expectation_values.json")


def test_real_wavefunction_io():
    wf = Wavefunction([0, 1, 0, 0, 0, 0, 0, 0])
    save_wavefunction(wf, "wavefunction.json")
    loaded_wf = load_wavefunction("wavefunction.json")
    assert np.allclose(wf.amplitudes, loaded_wf.amplitudes)
    remove_file_if_exists("wavefunction.json")


def test_imag_wavefunction_io():
    wf = Wavefunction([0, 1j, 0, 0, 0, 0, 0, 0])
    save_wavefunction(wf, "wavefunction.json")
    loaded_wf = load_wavefunction("wavefunction.json")
    assert np.allclose(wf.amplitudes, loaded_wf.amplitudes)
    remove_file_if_exists("wavefunction.json")


def test_sample_from_wavefunction():
    wavefunction = create_random_wavefunction(4)

    samples = sample_from_wavefunction(wavefunction, 10000)
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
        sampled_probabilities.append(sampled_dict[measurement] / 10000)

    probabilities = wavefunction.probabilities()
    for sampled_prob, exact_prob in zip(sampled_probabilities, probabilities):
        assert np.allclose(sampled_prob, exact_prob, atol=0.01)


def test_sample_from_wavefunction_column_vector():
    n_qubits = 4
    # NOTE: our indexing places the state of qubit i at the ith index of the tuple. Hence |01> will result in
    # the tuple (1, 0)
    expected_bitstring = (1, 0, 0, 0)
    amplitudes = np.array([0] * (2 ** n_qubits)).reshape(2 ** n_qubits, 1)
    amplitudes[1] = 1  # |0001> will be measured in all cases.
    wavefunction = Wavefunction(amplitudes)
    sample = set(sample_from_wavefunction(wavefunction, 500))
    assert len(sample) == 1
    assert sample.pop() == expected_bitstring


def test_sample_from_wavefunction_row_vector():
    n_qubits = 4
    # NOTE: our indexing places the state of qubit i at the ith index of the tuple. Hence |01> will result in
    # the tuple (1, 0)
    expected_bitstring = (1, 0, 0, 0)
    amplitudes = np.array([0] * (2 ** n_qubits))
    amplitudes[1] = 1  # |0001> will be measured in all cases.
    wavefunction = Wavefunction(amplitudes)
    sample = set(sample_from_wavefunction(wavefunction, 500))
    assert len(sample) == 1
    assert sample.pop() == expected_bitstring


def test_sample_from_wavefunction_list():
    n_qubits = 4
    # NOTE: our indexing places the state of qubit i at the ith index of the tuple. Hence |01> will result in
    # the tuple (1, 0)
    expected_bitstring = (1, 0, 0, 0)
    amplitudes = [0] * (2 ** n_qubits)
    amplitudes[1] = 1  # |0001> will be measured in all cases.
    wavefunction = Wavefunction(amplitudes)
    sample = set(sample_from_wavefunction(wavefunction, 500))
    assert len(sample) == 1
    assert sample.pop() == expected_bitstring


def test_parities_io():
    measurements = [(1, 0), (1, 0), (0, 1), (0, 0)]
    op = IsingOperator("[Z0] + [Z1] + [Z0 Z1]")
    parities = get_parities_from_measurements(measurements, op)
    save_parities(parities, "parities.json")
    loaded_parities = load_parities("parities.json")
    assert np.allclose(parities.values, loaded_parities.values)
    assert len(parities.correlations) == len(loaded_parities.correlations)
    for i in range(len(parities.correlations)):
        assert np.allclose(parities.correlations[i], loaded_parities.correlations[i])
    remove_file_if_exists("parities.json")


def test_get_expectation_values_from_parities():
    parities = Parities(values=np.array([[18, 50], [120, 113], [75, 26]]))
    expectation_values = get_expectation_values_from_parities(parities)

    assert len(expectation_values.values) == 3
    assert np.isclose(expectation_values.values[0], -0.47058823529411764)
    assert np.isclose(expectation_values.values[1], 0.030042918454935622)
    assert np.isclose(expectation_values.values[2], 0.48514851485148514)

    assert len(expectation_values.estimator_covariances) == 3
    assert np.allclose(
        expectation_values.estimator_covariances[0],
        np.array([[0.014705882352941176]]),
    )
    assert np.allclose(
        expectation_values.estimator_covariances[1], np.array([[0.00428797]])
    )

    assert np.allclose(
        expectation_values.estimator_covariances[2], np.array([[0.0075706]])
    )


def test_expectation_values_to_real():
    # Given
    expectation_values = ExpectationValues(np.array([0.0 + 0.1j, 0.0 + 1e-10j, -1.0]))
    target_expectation_values = ExpectationValues(np.array([0.0, 0.0, -1.0]))

    # When
    real_expectation_values = expectation_values_to_real(expectation_values)

    # Then
    for value in expectation_values.values:
        assert not isinstance(value, complex)
    np.testing.assert_array_equal(
        real_expectation_values.values, target_expectation_values.values
    )


def test_convert_bitstring_to_int():
    bitstring = (0, 1, 0, 1, 0, 1)
    assert convert_bitstring_to_int(bitstring) == 42


def test_check_parity_odd_string():
    bitstring = "01001"
    marked_qubits = (1, 2, 3)
    assert not check_parity(bitstring, marked_qubits)


def test_check_parity_even_string():
    bitstring = "01101"
    marked_qubits = (1, 2, 3)
    assert check_parity(bitstring, marked_qubits)


def test_check_parity_odd_tuple():
    bitstring = (0, 1, 0, 0, 1)
    marked_qubits = (1, 2, 3)
    assert not check_parity(bitstring, marked_qubits)


def test_check_parity_even_tuple():
    bitstring = (0, 1, 1, 0, 1)
    marked_qubits = (1, 2, 3)
    assert check_parity(bitstring, marked_qubits)


def test_get_expectation_value_from_frequencies():
    bitstrings = ["001", "001", "110", "000"]
    bitstring_frequencies = dict(Counter(bitstrings))
    marked_qubits = (1, 2)
    assert np.isclose(
        get_expectation_value_from_frequencies(marked_qubits, bitstring_frequencies),
        -0.5,
    )


def test_concatenate_expectation_values():
    expectation_values_set = [
        ExpectationValues(np.array([1.0, 2.0])),
        ExpectationValues(np.array([3.0, 4.0])),
    ]

    combined_expectation_values = concatenate_expectation_values(expectation_values_set)
    assert combined_expectation_values.correlations is None
    assert combined_expectation_values.estimator_covariances is None
    assert np.allclose(combined_expectation_values.values, [1.0, 2.0, 3.0, 4.0])


def test_concatenate_expectation_values_with_cov_and_corr():
    expectation_values_set = [
        ExpectationValues(
            np.array([1.0, 2.0]),
            estimator_covariances=[np.array([[0.1, 0.2], [0.3, 0.4]])],
            correlations=[np.array([[-0.1, -0.2], [-0.3, -0.4]])],
        ),
        ExpectationValues(
            np.array([3.0, 4.0]),
            estimator_covariances=[np.array([[0.1]]), np.array([[0.2]])],
            correlations=[np.array([[-0.1]]), np.array([[-0.2]])],
        ),
    ]
    combined_expectation_values = concatenate_expectation_values(expectation_values_set)
    assert len(combined_expectation_values.estimator_covariances) == 3
    assert np.allclose(
        combined_expectation_values.estimator_covariances[0],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    assert np.allclose(combined_expectation_values.estimator_covariances[1], [[0.1]])
    assert np.allclose(combined_expectation_values.estimator_covariances[2], [[0.2]])

    assert len(combined_expectation_values.correlations) == 3
    assert np.allclose(
        combined_expectation_values.correlations[0],
        [[-0.1, -0.2], [-0.3, -0.4]],
    )
    assert np.allclose(combined_expectation_values.correlations[1], [[-0.1]])
    assert np.allclose(combined_expectation_values.correlations[2], [[-0.2]])

    assert np.allclose(combined_expectation_values.values, [1.0, 2.0, 3.0, 4.0])


class TestMeasurements:
    def test_io(self):
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
        with open(output_filename) as f:
            output_data = json.load(f)
        assert measurements_data == output_data

        remove_file_if_exists(input_filename)
        remove_file_if_exists(output_filename)

    def test_save_for_numpy_integers(self):
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
        assert target_measurements.bitstrings == recreated_measurements.bitstrings
        remove_file_if_exists("measurementstest.json")

    def test_intialize_with_bitstrings(self):
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
        assert measurements.bitstrings == [
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

    def test_intialize_with_counts(self):
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
        assert measurements.bitstrings == [
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

    def test_bitstrings(self):
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
        assert measurements.bitstrings == [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (1, 0, 1),
            (0, 0, 1),
        ]

        remove_file_if_exists(input_filename)

    def test_get_counts(self):
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
        assert measurements_data["counts"] == counts

        remove_file_if_exists(input_filename)

    def test_get_distribution(self):
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
        assert distribution.distribution_dict == {
            "000": 1 / 9,
            "001": 2 / 9,
            "010": 1 / 9,
            "011": 1 / 9,
            "100": 1 / 9,
            "101": 1 / 9,
            "110": 1 / 9,
            "111": 1 / 9,
        }

        remove_file_if_exists(input_filename)

    def test_add_counts(self):
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
        assert measurements.bitstrings == [
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
        assert measurements.get_counts() == {
            "000": 1,
            "001": 2,
            "010": 1,
            "011": 1,
            "100": 1,
            "101": 1,
            "110": 1,
            "111": 1,
        }

    def test_add_measurements(self):
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
        assert measurements.bitstrings == [
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
        measurements.bitstrings += bitstrings

        # Then
        assert measurements.bitstrings == [
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
        ]

    def test_get_expectation_values_from_measurements(self):
        # Given
        measurements = Measurements(
            [(0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 1)]
        )
        ising_operator = IsingOperator("10[] + [Z0 Z1] - 15[Z1 Z2]")
        target_expectation_values = np.array([10, -0.2, -3])
        target_correlations = np.array([[100, -2, -30], [-2, 1, -9], [-30, -9, 225]])
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
        np.testing.assert_allclose(expectation_values.values, target_expectation_values)
        assert len(expectation_values.correlations) == 1
        np.testing.assert_allclose(
            expectation_values.correlations[0], target_correlations
        )
        assert len(expectation_values.estimator_covariances) == 1
        np.testing.assert_allclose(
            expectation_values.estimator_covariances[0], target_covariances
        )

    def test_get_expectation_values_from_measurements_with_bessel_correction(self):
        # Given
        measurements = Measurements(
            [(0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 1)]
        )
        ising_operator = IsingOperator("10[] + [Z0 Z1] - 15[Z1 Z2]")
        target_expectation_values = np.array([10, -0.2, -3])
        target_correlations = np.array([[100, -2, -30], [-2, 1, -9], [-30, -9, 225]])
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
        np.testing.assert_allclose(expectation_values.values, target_expectation_values)
        assert len(expectation_values.correlations) == 1
        np.testing.assert_allclose(
            expectation_values.correlations[0], target_correlations
        )
        assert len(expectation_values.estimator_covariances) == 1
        np.testing.assert_allclose(
            expectation_values.estimator_covariances[0], target_covariances
        )

    @pytest.mark.parametrize(
        "bitstring_distribution, number_of_samples",
        [
            (BitstringDistribution({"00": 0.5, "11": 0.5}), 1),
            (BitstringDistribution({"00": 0.5, "11": 0.5}), 10),
            (BitstringDistribution({"00": 0.5, "11": 0.5}), 51),
            (BitstringDistribution({"00": 0.5, "11": 0.5}), 137),
            (BitstringDistribution({"00": 0.5, "11": 0.5}), 5000),
            (BitstringDistribution({"0000": 0.137, "0001": 0.863}), 100),
            (
                BitstringDistribution(
                    {"00": 0.1234, "01": 0.5467, "10": 0.0023, "11": 0.3276}
                ),
                100,
            ),
            (
                BitstringDistribution(
                    {
                        "0000": 0.06835580857498666,
                        "1000": 0.060975627112613416,
                        "0100": 0.05976605586194627,
                        "1100": 0.07138587439957303,
                        "0010": 0.06474168297455969,
                        "1010": 0.0825036470378936,
                        "0110": 0.09861252446183953,
                        "1110": 0.0503013698630137,
                        "0001": 0.04496424123821384,
                        "1001": 0.07317221135029355,
                        "0101": 0.08171161714997331,
                        "1101": 0.03753940579967977,
                        "0011": 0.05157676570005337,
                        "1011": 0.05,
                        "0111": 0.04964419142501335,
                        "1111": 0.05474897705034692,
                    }
                ),
                5621,
            ),
        ],
    )
    def test_get_measurements_representing_distribution_returns_right_number_of_samples(
        self, bitstring_distribution, number_of_samples
    ):
        measurements = Measurements.get_measurements_representing_distribution(
            bitstring_distribution, number_of_samples
        )
        assert len(measurements.bitstrings) == number_of_samples

    @pytest.mark.parametrize(
        "bitstring_distribution, number_of_samples, expected_counts",
        [
            (
                BitstringDistribution({"01": 0.3333333, "11": (1 - 0.3333333)}),
                3,
                {"01": 1, "11": 2},
            ),
            (
                BitstringDistribution({"01": 0.9999999, "11": (1 - 0.9999999)}),
                1,
                {"01": 1},
            ),
        ],
    )
    def test_get_measurements_representing_distribution_correctly_samples_leftover_bitstrings(
        self, bitstring_distribution, number_of_samples, expected_counts
    ):
        random.seed(RNDSEED)
        measurements = Measurements.get_measurements_representing_distribution(
            bitstring_distribution, number_of_samples
        )
        assert measurements.get_counts() == expected_counts

    def test_get_measurements_representing_distribution_randomly_samples_leftover_bitstrings_when_probabilities_equal(
        self,
    ):
        random.seed(RNDSEED)
        bitstring_distribution = BitstringDistribution({"00": 0.5, "11": 0.5})
        number_of_samples = 51
        max_number_of_trials = 10
        got_different_measurements = False
        previous_measurements = Measurements.get_measurements_representing_distribution(
            bitstring_distribution, number_of_samples
        )

        while not got_different_measurements:
            measurements = Measurements.get_measurements_representing_distribution(
                bitstring_distribution, number_of_samples
            )

            assert (
                measurements.get_counts()
                == {
                    "00": 25,
                    "11": 26,
                }
                or measurements.get_counts() == {"00": 26, "11": 25}
            )

            if measurements.get_counts() != previous_measurements.get_counts():
                got_different_measurements = True

            max_number_of_trials -= 1
            if max_number_of_trials == 0:
                break
        assert got_different_measurements

    @pytest.mark.parametrize(
        "bitstring_distribution",
        [
            BitstringDistribution({"00": 0.5, "11": 0.5}),
            BitstringDistribution({"000": 0.5, "101": 0.5}),
            BitstringDistribution({"0000": 0.137, "0001": 0.863}),
            BitstringDistribution(
                {"00": 0.1234, "01": 0.5467, "10": 0.0023, "11": 0.3276}
            ),
        ],
    )
    def test_get_measurements_representing_distribution_gives_exactly_right_counts(
        self, bitstring_distribution
    ):
        number_of_samples = 10000
        measurements = Measurements.get_measurements_representing_distribution(
            bitstring_distribution, number_of_samples
        )

        counts = measurements.get_counts()
        for bitstring, probability in bitstring_distribution.distribution_dict.items():
            assert probability * number_of_samples == counts[bitstring]
