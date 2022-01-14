from json import load
from os import remove

import numpy as np
import pytest
from zquantum.core.circuits import CNOT, Circuit, H, X
from zquantum.core.distribution import MeasurementOutcomeDistribution
from zquantum.core.measurement import Measurements
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.core.trackers import MeasurementTrackingBackend


@pytest.fixture
def backend():
    backend = MeasurementTrackingBackend(SymbolicSimulator(), "tracker_test")
    backend.inner_backend.number_of_circuits_run = 0
    backend.inner_backend.number_of_jobs_run = 0
    return backend


class TestMeasurementTrackingBackend:
    """First we repeat tests from the QuantumBackendTests class
    modified to work with MeasurementTrackingBackend.

    Then we check that serialization performs as expected.
    """

    def test_run_circuit_and_measure_correct_indexing(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        n_samples = 100

        try:
            # When
            measurements = backend.run_circuit_and_measure(circuit, n_samples)

            # Then (since SPAM error could result in unexpected bitstrings, we make
            # sure the most common bitstring is the one we expect)
            counts = measurements.get_counts()
            assert max(counts, key=counts.get) == "001"
            assert backend.inner_backend.number_of_circuits_run == 1
            assert backend.inner_backend.number_of_jobs_run == 1
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    @pytest.mark.parametrize("n_samples", [-1, 0])
    def test_run_circuit_and_measure_fails_for_invalid_n_samples(
        self, backend, n_samples
    ):
        # Given
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        with pytest.raises(ValueError):
            backend.run_circuit_and_measure(circuit, n_samples)
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_run_circuitset_and_measure(self, backend):
        # Given
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        n_samples = 100
        number_of_circuits = 25

        # When
        n_samples_per_circuit = [n_samples] * number_of_circuits
        try:
            measurements_set = backend.run_circuitset_and_measure(
                [circuit] * number_of_circuits, n_samples_per_circuit
            )

            for measurements in measurements_set:
                counts = measurements.get_counts()
                assert max(counts, key=counts.get) == "001"
            assert backend.inner_backend.number_of_circuits_run == number_of_circuits

            if backend.inner_backend.supports_batching:
                assert backend.inner_backend.number_of_jobs_run == int(
                    np.ceil(number_of_circuits / backend.inner_backend.batch_size)
                )
            else:
                assert backend.inner_backend.number_of_jobs_run == number_of_circuits
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_run_circuitset_and_measure_n_samples(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        first_circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        second_circuit = Circuit([X(0), X(1), X(2)])
        n_samples_per_circuit = [100, 105]

        try:
            # When
            measurements_set = backend.run_circuitset_and_measure(
                [first_circuit, second_circuit], n_samples_per_circuit
            )

            # Then (since SPAM error could result in unexpected bitstrings, we make
            # sure the most common bitstring is the one we expect)
            counts = measurements_set[0].get_counts()
            assert max(counts, key=counts.get) == "001"
            counts = measurements_set[1].get_counts()
            assert max(counts, key=counts.get) == "111"

            assert len(measurements_set[0].bitstrings) == n_samples_per_circuit[0]
            assert len(measurements_set[1].bitstrings) == n_samples_per_circuit[1]

            assert backend.inner_backend.number_of_circuits_run == 2
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_get_bitstring_distribution(self, backend):
        # Given
        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])
        n_samples = 1000

        try:
            # When
            distribution = backend.get_bitstring_distribution(
                circuit, n_samples=n_samples
            )

            # Then
            assert isinstance(distribution, MeasurementOutcomeDistribution)
            assert distribution.get_number_of_subsystems() == 3
            assert distribution.distribution_dict[(0, 0, 0)] > 1 / 3
            assert distribution.distribution_dict[(1, 1, 1)] > 1 / 3
            assert backend.inner_backend.number_of_circuits_run == 1
            assert backend.inner_backend.number_of_jobs_run == 1
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_get_measurement_outcome_distribution(self, backend):
        # Given
        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])
        n_samples = 1000

        try:
            # When
            distribution = backend.get_measurement_outcome_distribution(
                circuit, n_samples=n_samples
            )

            # Then
            assert isinstance(distribution, MeasurementOutcomeDistribution)
            assert distribution.get_number_of_subsystems() == 3
            assert distribution.distribution_dict[(0, 0, 0)] > 1 / 3
            assert distribution.distribution_dict[(1, 1, 1)] > 1 / 3
            assert backend.inner_backend.number_of_circuits_run == 1
            assert backend.inner_backend.number_of_jobs_run == 1
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_serialization_of_measurement_data_from_circuit(self, backend):
        try:
            # When
            backend.run_circuit_and_measure(Circuit([X(0), X(0)]), n_samples=10)
            with open(backend.raw_data_file_name) as f:
                data = load(f)

            # Then
            assert data["raw-data"][0]["device"] == "SymbolicSimulator"
            assert data["raw-data"][0]["number_of_shots"] == 10
            assert data["raw-data"][0]["data_type"] == "measurement"
            """Assert solutions are in the recorded data"""
            assert {"0": 10} == data["raw-data"][0]["counts"]
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_serialization_of_measurement_data_from_circuitset(self, backend):
        # Given
        circuitset = [Circuit([X(0), X(0)]), Circuit([X(0)])]
        n_samples = [10] * 2

        try:
            # When
            backend.run_circuitset_and_measure(circuitset, n_samples=n_samples)
            f = open(backend.raw_data_file_name)
            data = load(f)

            # Then
            for i in range(1):
                assert data["raw-data"][i]["device"] == "SymbolicSimulator"
                assert data["raw-data"][i]["number_of_shots"] == 10
                assert data["raw-data"][i]["data_type"] == "measurement"
            """Assert solutions are in the recorded data"""
            assert {"0": 10} == data["raw-data"][0]["counts"]
            assert {"1": 10} == data["raw-data"][1]["counts"]
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_serialization_of_bitstrings(self, backend):
        # Given
        backend.record_bitstrings = True

        try:
            # When
            backend.run_circuit_and_measure(Circuit([X(0), X(0)]), n_samples=10)
            f = open(backend.raw_data_file_name)
            data = load(f)

            # Then
            assert data["raw-data"][0]["device"] == "SymbolicSimulator"
            assert data["raw-data"][0]["number_of_shots"] == 10
            """Assert solutions are in the recorded data"""
            assert [[0] for _ in range(10)] == data["raw-data"][0]["bitstrings"]
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_serialization_of_bitstring_distributions(self, backend):
        try:
            # When
            backend.get_bitstring_distribution(Circuit([X(0), X(0)]), n_samples=10)
            with open(backend.raw_data_file_name) as f:
                data = load(f)

            # Then
            assert data["raw-data"][0]["device"] == "SymbolicSimulator"
            assert data["raw-data"][0]["number_of_shots"] == 10
            """Assert solutions are in the recorded data"""
            assert (
                "BitstringDistribution(input={(0,): 1.0})"
                == data["raw-data"][0]["distribution"]
            )
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_serialization_of_measurement_outcome_distributions(self, backend):
        try:
            # When
            backend.get_measurement_outcome_distribution(Circuit([X(0), X(0)]), n_samples=10)
            with open(backend.raw_data_file_name) as f:
                data = load(f)

            # Then
            assert data["raw-data"][0]["device"] == "SymbolicSimulator"
            assert data["raw-data"][0]["number_of_shots"] == 10
            """Assert solutions are in the recorded data"""
            assert (
                "MeasurementOutcomeDistribution(input={(0,): 1.0})"
                == data["raw-data"][0]["distribution"]
            )
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)

    def test_serialization_of_multiple_trackers_with_same_backend(self, backend):
        # Given
        backend_2 = MeasurementTrackingBackend(backend.inner_backend, "tracker_test_2")

        try:
            # When
            backend.run_circuit_and_measure(Circuit([X(0), X(0)]), n_samples=10)
            backend_2.run_circuit_and_measure(Circuit([X(0)]), n_samples=10)
            with open(backend.raw_data_file_name) as f:
                data = load(f)
            with open(backend_2.raw_data_file_name) as f_2:
                data_2 = load(f_2)

            # Then
            assert data["raw-data"][0]["device"] == "SymbolicSimulator"
            assert data["raw-data"][0]["number_of_shots"] == 10
            assert data_2["raw-data"][0]["device"] == "SymbolicSimulator"
            assert data_2["raw-data"][0]["number_of_shots"] == 10
            """Assert solutions are in the recorded data"""
            assert {"0": 10} == data["raw-data"][0]["counts"]
            assert {"1": 10} == data_2["raw-data"][0]["counts"]
            assert backend.inner_backend.number_of_circuits_run == 2
            assert backend.inner_backend.number_of_jobs_run == 2
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)
            remove(backend_2.raw_data_file_name)

    def test_serialization_when_inner_backend_used_for_different_task(self, backend):
        # Given
        backend_2 = backend.inner_backend

        try:
            # When
            backend.run_circuit_and_measure(Circuit([X(0), X(0)]), n_samples=10)
            measurement = backend_2.run_circuit_and_measure(
                Circuit([X(0)]), n_samples=10
            )
            with open(backend.raw_data_file_name) as f:
                data = load(f)

            # Then
            assert data["raw-data"][0]["device"] == "SymbolicSimulator"
            assert data["raw-data"][0]["number_of_shots"] == 10
            """Assert solutions are in the recorded data"""
            assert {"0": 10} == data["raw-data"][0]["counts"]
            assert {"1": 10} == measurement.get_counts()
        finally:
            # Cleanup
            remove(backend.raw_data_file_name)
