import numpy as np
import pytest
from .cost_function import AnsatzBasedCostFunction
from .interfaces.mock_objects import MockQuantumSimulator, MockEstimator, MockAnsatz
from openfermion import QubitOperator


@pytest.fixture
def ansatz_based_cost_function():
    target_operator = QubitOperator("Z0")
    ansatz = MockAnsatz(number_of_layers=1, problem_size=1)
    backend = MockQuantumSimulator()
    estimator = MockEstimator()
    return AnsatzBasedCostFunction(
        target_operator, ansatz, backend, estimator=estimator
    )


def test_ansatz_based_cost_function_returns_value_between_plus_and_minus_one(
    ansatz_based_cost_function,
):
    params = np.array([1])
    value = ansatz_based_cost_function(params)
    assert -1 <= value <= 1


@pytest.fixture
def noisy_ansatz():
    target_operator = QubitOperator("Z0")
    ansatz = MockAnsatz(number_of_layers=2, problem_size=1)
    backend = MockQuantumSimulator()
    estimator = MockEstimator()
    return AnsatzBasedCostFunction(
        target_operator, ansatz, backend, estimator=estimator, parameter_precision=1e-4
    )


def test_ansatz_based_cost_function_adds_noise_to_parameters(noisy_ansatz):
    params = np.array([0.1, 2.3])
    value1 = noisy_ansatz(params)
    value2 = noisy_ansatz(params)
    assert value1 != value2

