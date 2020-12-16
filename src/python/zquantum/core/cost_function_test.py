import numpy as np
import pytest
from unittest import mock
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
    ansatz.get_executable_circuit = mock.Mock(wraps=ansatz.get_executable_circuit)
    return AnsatzBasedCostFunction(
        target_operator,
        ansatz,
        backend,
        estimator=estimator,
        parameter_precision=1e-4,
        parameter_precision_seed=1234
    )


def test_ansatz_based_cost_function_adds_noise_to_parameters(noisy_ansatz):
    generator = np.random.default_rng(1234)

    # We expect the below to get added to parameters
    noise = generator.normal(0, 1e-4, 2)

    params = np.array([0.1, 2.3])

    # ansatz based cost function may modify params in place
    # and we need original ones - therefore we pass a copy
    noisy_ansatz(np.array(params))

    # We only called our function once, therefore the following should be true
    noisy_ansatz.ansatz.get_executable_circuit.assert_called_once()

    # and if only everything went right, this only call should be of the form
    # noisy_ansatz.ansatz.get_executable_circuit(params+noise)
    # Therefore, we extract the single argument and compare it to the
    # expected one.
    assert np.array_equal(
        noisy_ansatz.ansatz.get_executable_circuit.call_args[0][0],
        noise+params
    )

    # Note, normally, we weould just do it in a single assert:
    # noisy_ansatz.ansatz.get_executable_circuit.assert_called_once_with(params_noise)
    # However, this does not work with numpy arrays, as it uses == operator
    # to compare arguments, which does not produce boolean value for numpy arrays

