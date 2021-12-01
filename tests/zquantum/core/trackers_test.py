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
    return MeasurementTrackingBackend(SymbolicSimulator())


class TestMeasurementTrackingBackend:
    """First we repeat tests from the QuantumBackendTests class
    modified to work with a measurement tracking backend.

    Then we check to make sure that the json files produced are
    as expected.
    """

    def test_run_circuit_and_measure_correct_indexing(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        backend.inner_backend.number_of_circuits_run = 0
        backend.inner_backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        n_samples = 100
        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples)

        # Then (since SPAM error could result in unexpected bitstrings, we make sure the
        # most common bitstring is the one we expect)
        counts = measurements.get_counts()
        assert max(counts, key=counts.get) == "001"
        assert backend.inner_backend.number_of_circuits_run == 1
        assert backend.inner_backend.number_of_jobs_run == 1
        # Cleanup
        remove(backend.file_name)

    @pytest.mark.parametrize("n_samples", [-1, 0, 100.2, 1000.0])
    def test_run_circuit_and_measure_fails_for_invalid_n_samples(
        self, backend, n_samples
    ):
        # Given
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        with pytest.raises(AssertionError):
            backend.run_circuit_and_measure(circuit, n_samples)
            # Cleanup
            remove(backend.file_name)

    @pytest.mark.parametrize("n_samples", [1, 2, 10, 100])
    def test_run_circuit_and_measure_correct_num_measurements_attribute(
        self, backend, n_samples
    ):
        # Given
        backend.inner_backend.number_of_circuits_run = 0
        backend.inner_backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples)

        # Then
        assert isinstance(measurements, Measurements)
        assert len(measurements.bitstrings) == n_samples
        assert backend.inner_backend.number_of_circuits_run == 1
        assert backend.inner_backend.number_of_jobs_run == 1
        # Cleanup
        remove(backend.file_name)

    @pytest.mark.parametrize("n_samples", [1, 2, 10, 100])
    def test_run_circuit_and_measure_correct_num_measurements_argument(
        self, backend, n_samples
    ):
        # Given
        backend.inner_backend.number_of_circuits_run = 0
        backend.inner_backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples)

        # Then
        assert isinstance(measurements, Measurements)
        assert len(measurements.bitstrings) == n_samples
        assert backend.inner_backend.number_of_circuits_run == 1
        assert backend.inner_backend.number_of_jobs_run == 1
        # Cleanup
        remove(backend.file_name)

    def test_if_all_measurements_have_the_same_number_of_bits(self, backend):
        # Given
        backend.inner_backend.number_of_circuits_run = 0
        backend.inner_backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])

        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples=100)

        # Then
        assert all(len(bitstring) == 3 for bitstring in measurements.bitstrings)
        assert backend.inner_backend.number_of_circuits_run == 1
        assert backend.inner_backend.number_of_jobs_run == 1
        # Cleanup
        remove(backend.file_name)

    def test_run_circuitset_and_measure(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        backend.inner_backend.number_of_circuits_run = 0
        backend.inner_backend.number_of_jobs_run = 0
        circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        n_samples = 100
        number_of_circuits = 25
        # When
        n_samples_per_circuit = [n_samples] * number_of_circuits
        measurements_set = backend.run_circuitset_and_measure(
            [circuit] * number_of_circuits, n_samples_per_circuit
        )

        # Then (since SPAM error could result in unexpected bitstrings, we make sure the
        # most common bitstring is the one we expect)
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
        # Cleanup
        remove(backend.file_name)

    def test_run_circuitset_and_measure_n_samples(self, backend):
        # Note: this test may fail with noisy devices
        # Given
        backend.inner_backend.number_of_circuits_run = 0
        backend.inner_backend.number_of_jobs_run = 0
        first_circuit = Circuit([X(0), X(0), X(1), X(1), X(2)])
        second_circuit = Circuit([X(0), X(1), X(2)])
        n_samples_per_circuit = [100, 105]

        # When
        measurements_set = backend.run_circuitset_and_measure(
            [first_circuit, second_circuit], n_samples_per_circuit
        )

        # Then (since SPAM error could result in unexpected bitstrings, we make sure the
        # most common bitstring is the one we expect)
        counts = measurements_set[0].get_counts()
        assert max(counts, key=counts.get) == "001"
        counts = measurements_set[1].get_counts()
        assert max(counts, key=counts.get) == "111"

        assert len(measurements_set[0].bitstrings) == n_samples_per_circuit[0]
        assert len(measurements_set[1].bitstrings) == n_samples_per_circuit[1]

        assert backend.inner_backend.number_of_circuits_run == 2
        # Cleanup
        remove(backend.file_name)

    def test_get_bitstring_distribution(self, backend):
        # Given
        backend.inner_backend.number_of_circuits_run = 0
        backend.inner_backend.number_of_jobs_run = 0
        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])
        n_samples = 1000

        # When
        distribution = backend.get_bitstring_distribution(circuit, n_samples=n_samples)

        # Then
        assert isinstance(distribution, MeasurementOutcomeDistribution)
        assert distribution.get_number_of_subsystems() == 3
        assert distribution.distribution_dict[(0, 0, 0)] > 1 / 3
        assert distribution.distribution_dict[(1, 1, 1)] > 1 / 3
        assert backend.inner_backend.number_of_circuits_run == 1
        assert backend.inner_backend.number_of_jobs_run == 1
        # Cleanup
        remove(backend.file_name)
