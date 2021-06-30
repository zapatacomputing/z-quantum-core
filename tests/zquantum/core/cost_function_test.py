from functools import partial
from unittest import mock

import numpy as np
import pytest
from openfermion import QubitOperator
from sympy import Symbol
from zquantum.core.cost_function import (
    AnsatzBasedCostFunction,
    get_ground_state_cost_function,
    sum_expectation_values,
)
from zquantum.core.estimation import (
    allocate_shots_proportionally,
    allocate_shots_uniformly,
    calculate_exact_expectation_values,
    estimate_expectation_values_by_averaging,
)
from zquantum.core.interfaces.mock_objects import MockAnsatz
from zquantum.core.measurement import ExpectationValues
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.core.utils import create_symbols_map

RNGSEED = 1234


@pytest.fixture(
    params=[
        {
            "target_operator": QubitOperator("Z0"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=1, problem_size=1
            ).parametrized_circuit,
            "backend": SymbolicSimulator(),
            "estimation_method": estimate_expectation_values_by_averaging,
            "estimation_preprocessors": [
                partial(allocate_shots_uniformly, number_of_shots=1)
            ],
        },
        {
            "target_operator": QubitOperator("Z0 Z1"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=1, problem_size=2
            ).parametrized_circuit,
            "backend": SymbolicSimulator(),
            "estimation_method": estimate_expectation_values_by_averaging,
            "estimation_preprocessors": [
                partial(allocate_shots_uniformly, number_of_shots=1)
            ],
        },
        {
            "target_operator": QubitOperator("Z0 Z1"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=2, problem_size=2
            ).parametrized_circuit,
            "backend": SymbolicSimulator(),
            "estimation_method": estimate_expectation_values_by_averaging,
            "fixed_parameters": [1.2],
            "estimation_preprocessors": [
                partial(allocate_shots_uniformly, number_of_shots=1)
            ],
        },
        {
            "target_operator": QubitOperator("Z0 Z1"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=2, problem_size=2
            ).parametrized_circuit,
            "backend": SymbolicSimulator(),
            "estimation_method": estimate_expectation_values_by_averaging,
            "fixed_parameters": [1.2],
            "parameter_precision": 0.001,
            "parameter_precision_seed": RNGSEED,
            "estimation_preprocessors": [
                partial(allocate_shots_uniformly, number_of_shots=1)
            ],
        },
        {
            "target_operator": QubitOperator("Z0"),
            "parametrized_circuit": MockAnsatz(
                number_of_layers=1, problem_size=1
            ).parametrized_circuit,
            "backend": SymbolicSimulator(),
            "estimation_method": estimate_expectation_values_by_averaging,
            "estimation_preprocessors": [
                partial(allocate_shots_proportionally, total_n_shots=1)
            ],
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
    parametrized_circuit.bind = mock.Mock(wraps=parametrized_circuit.bind)
    backend = SymbolicSimulator()
    estimation_method = estimate_expectation_values_by_averaging
    estimation_preprocessors = [partial(allocate_shots_uniformly, number_of_shots=1)]
    noisy_ground_state_cost_function = get_ground_state_cost_function(
        target_operator,
        parametrized_circuit,
        backend,
        estimation_method=estimation_method,
        estimation_preprocessors=estimation_preprocessors,
        parameter_precision=1e-4,
        parameter_precision_seed=RNGSEED,
    )

    generator = np.random.default_rng(RNGSEED)

    # We expect the below to get added to parameters
    noise = generator.normal(0, 1e-4, 2)

    params = np.array([0.1, 2.3], dtype=float)

    expected_symbols_map = {
        Symbol("theta_0"): noise[0] + params[0],
        Symbol("theta_1"): noise[1] + params[1],
    }

    # ansatz based cost function may modify params in place
    # and we need original ones - therefore we pass a copy
    noisy_ground_state_cost_function(np.array(params))

    # We only called our function once, therefore the following should be true
    parametrized_circuit.bind.assert_called_with(expected_symbols_map)

    # and if only everything went right, this only call should be of the form
    # noisy_ansatz.ansatz.get_executable_circuit(params+noise)
    # Therefore, we extract the single argument and compare it to the
    # expected one.
    assert np.array_equal(
        parametrized_circuit.bind.call_args[0][0], expected_symbols_map
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
    backend = SymbolicSimulator()
    estimation_method = estimate_expectation_values_by_averaging
    estimation_preprocessors = [partial(allocate_shots_uniformly, number_of_shots=1)]
    return AnsatzBasedCostFunction(
        target_operator,
        ansatz,
        backend,
        estimation_method=estimation_method,
        estimation_preprocessors=estimation_preprocessors,
    )


def test_ansatz_based_cost_function_returns_value_between_plus_and_minus_one(
    ansatz_based_cost_function,
):
    params = np.array([1])
    value = ansatz_based_cost_function(params)
    assert -1 <= value <= 1


@pytest.fixture
def noisy_ansatz_cost_function_with_ansatz():
    target_operator = QubitOperator("Z0")
    ansatz = MockAnsatz(number_of_layers=2, problem_size=1)
    backend = SymbolicSimulator()
    estimation_method = mock.Mock(wraps=calculate_exact_expectation_values)
    return (
        AnsatzBasedCostFunction(
            target_operator,
            ansatz,
            backend,
            estimation_method=estimation_method,
            parameter_precision=1e-4,
            parameter_precision_seed=RNGSEED,
        ),
        ansatz,
    )


def test_ansatz_based_cost_function_adds_noise_to_parameters(
    noisy_ansatz_cost_function_with_ansatz,
):
    noisy_ansatz_cost_function = noisy_ansatz_cost_function_with_ansatz[0]
    ansatz = noisy_ansatz_cost_function_with_ansatz[1]
    generator = np.random.default_rng(RNGSEED)

    # We expect the below to get added to parameters
    noise = generator.normal(0, 1e-4, 2)

    params = np.array([0.1, 2.3])

    # ansatz based cost function may modify params in place
    # and we need original ones - therefore we pass a copy
    noisy_ansatz_cost_function(np.array(params))

    # We only called our function once, therefore the following should be true
    noisy_ansatz_cost_function.estimation_method.assert_called_once()

    # Here, we make the expected executable circuit with the noisy parameters
    noisy_symbols_map = create_symbols_map(
        ansatz.parametrized_circuit.free_symbols, noise + params
    )
    expected_noisy_circuit = ansatz.parametrized_circuit.bind(noisy_symbols_map)

    assert (
        noisy_ansatz_cost_function.estimation_method.call_args[0][1][0].circuit
        == expected_noisy_circuit
    )
