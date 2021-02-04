import numpy as np
import pytest
from unittest import mock
from sympy import Symbol
from .cost_function import (
    AnsatzBasedCostFunction,
    get_ground_state_cost_function,
    sum_expectation_values,
)
from .interfaces.mock_objects import MockQuantumSimulator, MockEstimator, MockAnsatz
from openfermion import QubitOperator
from .measurement import ExpectationValues

RNGSEED = 1234


@pytest.fixture(
    params=[
        {
            "target_operator": QubitOperator("Z0"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=1, problem_size=1
            ).parametrized_circuit,
            "backend": MockQuantumSimulator(),
            "estimator": MockEstimator(),
        },
        {
            "target_operator": QubitOperator("Z0 Z1"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=1, problem_size=2
            ).parametrized_circuit,
            "backend": MockQuantumSimulator(),
            "estimator": MockEstimator(),
        },
        {
            "target_operator": QubitOperator("Z0 Z1"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=2, problem_size=2
            ).parametrized_circuit,
            "backend": MockQuantumSimulator(),
            "estimator": MockEstimator(),
            "fixed_parameters": [1.2],
        },
        {
            "target_operator": QubitOperator("Z0 Z1"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=2, problem_size=2
            ).parametrized_circuit,
            "backend": MockQuantumSimulator(),
            "estimator": MockEstimator(),
            "fixed_parameters": [1.2],
            "parameter_precision": 0.001,
            "parameter_precision_seed": RNGSEED,
        },
    ]
)
def ground_state_cost_function(request):
    return get_ground_state_cost_function(**request.param)


def test_ground_state_cost_function_returns_value_between_plus_and_minus_one(
    ground_state_cost_function,
):
    params = np.array([1.0], dtype=float)
    value = ground_state_cost_function(params)
    assert -1 <= value <= 1


def test_noisy_ground_state_cost_function_adds_noise_to_parameters():
    target_operator = QubitOperator("Z0")
    parametrized_circuit = MockAnsatz(
        number_of_layers=2, problem_size=1
    ).parametrized_circuit
    parametrized_circuit.evaluate = mock.Mock(wraps=parametrized_circuit.evaluate)
    backend = MockQuantumSimulator()
    estimator = MockEstimator()
    noisy_ground_state_cost_function = get_ground_state_cost_function(
        target_operator,
        parametrized_circuit,
        backend,
        estimator=estimator,
        parameter_precision=1e-4,
        parameter_precision_seed=RNGSEED,
    )

    generator = np.random.default_rng(RNGSEED)

    # We expect the below to get added to parameters
    noise = generator.normal(0, 1e-4, 2)

    params = np.array([0.1, 2.3], dtype=float)

    expected_symbols_map = [
        (Symbol("theta_0"), noise[0] + params[0]),
        (Symbol("theta_1"), noise[1] + params[1]),
    ]

    # ansatz based cost function may modify params in place
    # and we need original ones - therefore we pass a copy
    noisy_ground_state_cost_function(np.array(params))

    # We only called our function once, therefore the following should be true
    parametrized_circuit.evaluate.assert_called_with(expected_symbols_map)

    # and if only everything went right, this only call should be of the form
    # noisy_ansatz.ansatz.get_executable_circuit(params+noise)
    # Therefore, we extract the single argument and compare it to the
    # expected one.
    assert np.array_equal(
        parametrized_circuit.evaluate.call_args[0][0], expected_symbols_map
    )

    # Note, normally, we weould just do it in a single assert:
    # noisy_ansatz.ansatz.get_executable_circuit.assert_called_once_with(params_noise)
    # However, this does not work with numpy arrays, as it uses == operator
    # to compare arguments, which does not produce boolean value for numpy arrays


def test_sum_expectation_values():
    expectation_values = ExpectationValues(np.array([5, -2, 1]))
    total = sum_expectation_values(expectation_values)
    assert np.isclose(total.value, 4)
    assert total.precision is None


def test_sum_expectation_values_with_covariances():
    values = np.array([5, -2, 1])
    correlations = [np.array([[1, 0.5], [0.5, 2]]), np.array([[7]])]
    covariances = [correlations[0] / 10, correlations[1] / 10]
    expectation_values = ExpectationValues(values, correlations, covariances)
    total = sum_expectation_values(expectation_values)
    assert np.isclose(total.value, 4)
    assert np.isclose(total.precision, np.sqrt((1 + 0.5 + 0.5 + 2 + 7) / 10))


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
        parameter_precision_seed=RNGSEED,
    )


def test_ansatz_based_cost_function_adds_noise_to_parameters(noisy_ansatz):
    generator = np.random.default_rng(RNGSEED)

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
        noisy_ansatz.ansatz.get_executable_circuit.call_args[0][0], noise + params
    )

    # Note, normally, we weould just do it in a single assert:
    # noisy_ansatz.ansatz.get_executable_circuit.assert_called_once_with(params_noise)
    # However, this does not work with numpy arrays, as it uses == operator
    # to compare arguments, which does not produce boolean value for numpy arrays
