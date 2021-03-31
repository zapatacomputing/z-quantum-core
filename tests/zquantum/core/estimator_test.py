from pyquil import Program
from pyquil.gates import X
from openfermion import QubitOperator, qubit_operator_sparse, IsingOperator
import numpy as np
import pytest

from .interfaces.estimator_test import EstimatorTests
from .interfaces.mock_objects import MockQuantumBackend, MockQuantumSimulator
from .estimator import (
    BasicEstimator,
    ExactEstimator,
    get_context_selection_circuit,
    get_context_selection_circuit_for_group,
)
from .measurement import ExpectationValues
from .circuit import Circuit


class TestEstimatorUtils:
    def test_get_context_selection_circuit_offdiagonal(self):
        term = ((0, "X"), (1, "Y"))
        circuit, ising_operator = get_context_selection_circuit(term)

        # Need to convert to QubitOperator in order to get matrix representation
        qubit_operator = QubitOperator()
        for ising_term in ising_operator.terms:
            qubit_operator += QubitOperator(
                ising_term, ising_operator.terms[ising_term]
            )

        target_unitary = qubit_operator_sparse(QubitOperator(term))
        transformed_unitary = (
            circuit.to_unitary().conj().T
            @ qubit_operator_sparse(qubit_operator)
            @ circuit.to_unitary()
        )

        assert np.allclose(target_unitary.todense(), transformed_unitary)

    def test_get_context_selection_circuit_diagonal(self):
        term = ((4, "Z"), (2, "Z"))
        circuit, ising_operator = get_context_selection_circuit(term)
        assert len(circuit.gates) == 0
        assert ising_operator == IsingOperator(term)

    def test_get_context_selection_circuit_for_group(self):
        group = QubitOperator(((0, "X"), (1, "Y"))) - 0.5 * QubitOperator(((1, "Y"),))
        circuit, ising_operator = get_context_selection_circuit_for_group(group)

        # Need to convert to QubitOperator in order to get matrix representation
        qubit_operator = QubitOperator()
        for ising_term in ising_operator.terms:
            qubit_operator += QubitOperator(
                ising_term, ising_operator.terms[ising_term]
            )

        target_unitary = qubit_operator_sparse(group)
        transformed_unitary = (
            circuit.to_unitary().conj().T
            @ qubit_operator_sparse(qubit_operator)
            @ circuit.to_unitary()
        )

        assert np.allclose(target_unitary.todense(), transformed_unitary)


class TestBasicEstimator(EstimatorTests):
    @pytest.fixture()
    def estimator(self, request):
        return BasicEstimator()

    @pytest.fixture()
    def target_operator(self, request):
        return QubitOperator("Z0")

    @pytest.fixture()
    def circuit(self, request):
        return Circuit(Program(X(0)))

    @pytest.fixture()
    def backend(self, request):
        return MockQuantumBackend(n_samples=20)

    @pytest.fixture()
    def n_samples(self, request):
        return 10

    def test_get_estimated_expectation_values(
        self, estimator, backend, circuit, target_operator, n_samples
    ):
        # When
        values = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=target_operator,
            n_samples=n_samples,
        ).values
        value = values[0]
        # Then
        assert len(values) == 1
        assert value >= -1
        assert value <= 1

    def test_get_estimated_expectation_values_samples_from_backend(
        self,
        estimator,
        backend,
        circuit,
        target_operator,
    ):
        # Given
        # When
        values = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=target_operator,
        ).values
        value = values[0]
        # Then
        assert len(values) == 1
        assert value >= -1
        assert value <= 1

    def test_n_samples_is_restored(self, estimator, backend, circuit, target_operator):
        # Given
        backend.n_samples = 5
        # When
        values = estimator.get_estimated_expectation_values(
            backend, circuit, target_operator, n_samples=10
        )
        # Then
        assert backend.n_samples == 5

    def test_get_estimated_expectation_values_with_constant(
        self, estimator, backend, circuit, n_samples
    ):
        # Given
        coefficient = -2
        constant_qubit_operator = QubitOperator((), coefficient) + QubitOperator(
            (0, "X")
        )

        # When
        values = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=constant_qubit_operator,
            n_samples=n_samples,
        ).values
        value = values[1]
        # Then
        assert len(values) == 2
        assert coefficient == value

    def test_get_estimated_expectation_values_optimal_shot_allocation(
        self, estimator, backend, circuit, target_operator
    ):
        # TODO: After a deterministic testing backend is imlemented, this test
        # should be updated to actually check that shots are being correctly
        # allocated and the expectation values correctly estimated.

        # Given
        # When
        values = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=target_operator,
            shot_allocation_strategy="optimal",
            n_total_samples=100,
        ).values
        value = values[0]
        # Then
        assert len(values) == 1
        assert value >= -1
        assert value <= 1

    def test_get_estimated_expectation_values_optimal_shot_allocation_with_prior(
        self, estimator, backend, circuit, target_operator
    ):
        # TODO: After a deterministic testing backend is imlemented, this test
        # should be updated to actually check that shots are being correctly
        # allocated and the expectation values correctly estimated.

        # Given
        # When
        estimator.prior_expectation_values = ExpectationValues(
            np.array([0 for _ in target_operator.terms])
        )
        values = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=target_operator,
            shot_allocation_strategy="optimal",
            n_total_samples=100,
        ).values
        value = values[0]
        # Then
        assert len(values) == 1
        assert value >= -1
        assert value <= 1

    @pytest.mark.parametrize(
        "n_samples,n_total_samples,shot_allocation_strategy",
        [
            (None, 100, "uniform"),
            (100, None, "optimal"),
            (100, 100, "optimal"),
            (100, 100, "uniform"),
            (100, None, "foo"),
        ],
    )
    def test_get_estimated_expectation_values_invalid_options(
        self,
        estimator,
        backend,
        circuit,
        target_operator,
        n_samples,
        n_total_samples,
        shot_allocation_strategy,
    ):
        with pytest.raises(ValueError):
            estimator.get_estimated_expectation_values(
                backend=backend,
                circuit=circuit,
                target_operator=target_operator,
                shot_allocation_strategy=shot_allocation_strategy,
                n_total_samples=n_total_samples,
                n_samples=n_samples,
            )


class TestExactEstimator(EstimatorTests):
    @pytest.fixture()
    def estimator(self, request):
        return ExactEstimator()

    @pytest.fixture()
    def target_operator(self, request):
        return QubitOperator("Z0")

    @pytest.fixture()
    def circuit(self, request):
        return Circuit(Program(X(0)))

    @pytest.fixture()
    def backend(self, request):
        return MockQuantumSimulator()

    @pytest.fixture()
    def n_samples(self, request):
        return None

    def test_require_quantum_simulator(
        self, estimator, backend, circuit, target_operator
    ):
        backend = MockQuantumBackend()
        with pytest.raises(AttributeError):
            value = estimator.get_estimated_expectation_values(
                backend=backend,
                circuit=circuit,
                target_operator=target_operator,
            ).values

    def test_get_estimated_expectation_values(
        self, estimator, backend, circuit, target_operator
    ):
        # Given
        # When
        values = estimator.get_estimated_expectation_values(
            backend=backend,
            circuit=circuit,
            target_operator=target_operator,
            n_samples=None,
        ).values
        value = values[0]
        # Then
        assert len(values) == 1
        assert value >= -1
        assert value <= 1
