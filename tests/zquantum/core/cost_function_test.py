from functools import partial
from unittest import mock

import numpy as np
import pytest
from openfermion import QubitOperator
from sympy import Symbol
from zquantum.core.cost_function import (
    AnsatzBasedCostFunction,
    CostFunction,
    EstimationTasksFactory,
    ParameterPreprocessor,
    add_noise,
    create_cost_function,
    fix_parameters,
    get_ground_state_cost_function,
    substitution_based_estimation_tasks_factory,
    sum_expectation_values,
)
from zquantum.core.estimation import (
    allocate_shots_uniformly,
    calculate_exact_expectation_values,
    estimate_expectation_values_by_averaging,
)
from zquantum.core.interfaces.mock_objects import MockAnsatz
from zquantum.core.measurement import ExpectationValues
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.core.utils import create_symbols_map

RNGSEED = 1234

BACKEND = SymbolicSimulator()
ESTIMATION_METHOD = estimate_expectation_values_by_averaging
ESTIMATION_PREPROCESSORS = [partial(allocate_shots_uniformly, number_of_shots=1)]


class TestGroundStateCostFunction:
    @pytest.fixture(
        params=[
            {
                "target_operator": QubitOperator("Z0"),
                "parametrized_circuit": MockAnsatz(
                    number_of_layers=1, problem_size=1
                ).parametrized_circuit,
            },
            {
                "target_operator": QubitOperator("Z0 Z1"),
                "parametrized_circuit": MockAnsatz(
                    number_of_layers=1, problem_size=2
                ).parametrized_circuit,
            },
            {
                "target_operator": QubitOperator("Z0 Z1"),
                "parametrized_circuit": MockAnsatz(
                    number_of_layers=2, problem_size=2
                ).parametrized_circuit,
                "fixed_parameters": [1.2],
            },
            {
                "target_operator": QubitOperator("Z0 Z1"),
                "parametrized_circuit": MockAnsatz(
                    number_of_layers=2, problem_size=2
                ).parametrized_circuit,
                "fixed_parameters": [1.2],
                "parameter_precision": 0.001,
                "parameter_precision_seed": RNGSEED,
            },
            {
                "target_operator": QubitOperator("Z0"),
                "parametrized_circuit": MockAnsatz(
                    number_of_layers=1, problem_size=1
                ).parametrized_circuit,
            },
        ]
    )
    def ground_state_cost_function(self, request):
        return get_ground_state_cost_function(
            **request.param,
            backend=BACKEND,
            estimation_method=ESTIMATION_METHOD,
            estimation_preprocessors=ESTIMATION_PREPROCESSORS
        )

    def test_returns_value_between_plus_and_minus_one(
        self,
        ground_state_cost_function,
    ):
        params = np.array([1.0], dtype=float)
        value = ground_state_cost_function(params)
        assert -1 <= value <= 1

    def test_adds_noise_to_parameters(self):
        target_operator = QubitOperator("Z0")
        parametrized_circuit = MockAnsatz(
            number_of_layers=2, problem_size=1
        ).parametrized_circuit
        parametrized_circuit.bind = mock.Mock(wraps=parametrized_circuit.bind)
        noisy_ground_state_cost_function = get_ground_state_cost_function(
            target_operator,
            parametrized_circuit,
            BACKEND,
            ESTIMATION_METHOD,
            ESTIMATION_PREPROCESSORS,
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

        # Note, normally, we would just do it in a single assert:
        # noisy_ansatz.ansatz.get_executable_circuit.assert_called_once_with(params_noise)
        # However, this does not work with numpy arrays, as it uses == operator
        # to compare arguments, which does not produce boolean value for numpy arrays


class TestSumExpectationValues:
    def test_sum_expectation_values(self):
        expectation_values = ExpectationValues(np.array([5, -2, 1]))
        total = sum_expectation_values(expectation_values)
        assert np.isclose(total.value, 4)
        assert total.precision is None

    def test_sum_expectation_values_with_covariances(self):
        values = np.array([5, -2, 1])
        correlations = [np.array([[1, 0.5], [0.5, 2]]), np.array([[7]])]
        covariances = [correlations[0] / 10, correlations[1] / 10]
        expectation_values = ExpectationValues(values, correlations, covariances)
        total = sum_expectation_values(expectation_values)
        assert np.isclose(total.value, 4)
        assert np.isclose(total.precision, np.sqrt((1 + 0.5 + 0.5 + 2 + 7) / 10))


TARGET_OPERATOR = QubitOperator("Z0")
ANSATZ = MockAnsatz(number_of_layers=1, problem_size=1)


class TestAnsatzBasedCostFunction:
    @pytest.fixture()
    def old_ansatz_based_cost_function(self):
        return AnsatzBasedCostFunction(
            TARGET_OPERATOR,
            ANSATZ,
            BACKEND,
            ESTIMATION_METHOD,
            ESTIMATION_PREPROCESSORS,
        )

    @pytest.fixture()
    def ansatz_based_cost_function(self):
        estimation_factory = substitution_based_estimation_tasks_factory(
            TARGET_OPERATOR, ANSATZ, ESTIMATION_PREPROCESSORS
        )

        return create_cost_function(
            BACKEND,
            estimation_factory,
            ESTIMATION_METHOD,
        )

    def test_ansatz_based_cost_function_returns_value_between_plus_and_minus_one(
        self, ansatz_based_cost_function, old_ansatz_based_cost_function
    ):
        params = np.array([1])

        value = ansatz_based_cost_function(params)
        assert -1 <= value <= 1

        value = old_ansatz_based_cost_function(params)
        assert -1 <= value <= 1

    @pytest.fixture()
    def noisy_ansatz_cost_function_with_ansatz(self):
        ansatz = MockAnsatz(number_of_layers=2, problem_size=1)
        estimation_method = mock.Mock(wraps=calculate_exact_expectation_values)
        return (
            AnsatzBasedCostFunction(
                TARGET_OPERATOR,
                ansatz,
                BACKEND,
                estimation_method,
                parameter_precision=1e-4,
                parameter_precision_seed=RNGSEED,
            ),
            ansatz,
        )

    def test_ansatz_based_cost_function_adds_noise_to_parameters(
        self, noisy_ansatz_cost_function_with_ansatz
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


class TestFixParametersPreprocessor:
    def test_concatenates_params(self):
        preprocessor = fix_parameters(np.array([1.0, 2.0, 3.0]))
        params = np.array([0.5, 0.0, -1.0, np.pi])

        new_params = preprocessor(params)

        np.testing.assert_array_equal(
            new_params, [0.5, 0.0, -1.0, np.pi, 1.0, 2.0, 3.0]
        )

    def test_does_not_mutate_parameters(self):
        preprocessor = fix_parameters(np.array([-1.5, 2.0]))
        params = np.array([0.1, 0.2])

        preprocessor(params)

        np.testing.assert_array_equal(params, [0.1, 0.2])


class TestAddNoisePreprocessor:
    def test_correctly_seeds_rng(self):
        preprocessor_1 = add_noise(1e-5, RNGSEED)
        preprocessor_2 = add_noise(1e-5, RNGSEED)

        params = np.linspace(0, np.pi, 10)

        np.testing.assert_array_equal(preprocessor_1(params), preprocessor_2(params))

    def test_seeds_rng_during_initialization(self):
        preprocessor = add_noise(1e-4, RNGSEED)
        params = np.array([0.1, 0.2, -0.5])

        # The second call to preprocessor should advance generator if it was
        # seeded only during initialization, hence the second call should produce
        # different result.
        assert not np.array_equal(preprocessor(params), preprocessor(params))

    def test_mean_of_added_noise_is_correct(self):
        preprocessor = add_noise(0.001, RNGSEED)
        num_params = 100
        num_repetitions = 10
        params = np.ones(num_params)

        average_diff = sum(
            (params - preprocessor(params)).sum() for _ in range(num_repetitions)
        ) / (num_repetitions * num_params)

        np.testing.assert_allclose(average_diff, 0.0, atol=1e-03)

    def test_std_of_added_noise_is_correct(self):
        preprocessor = add_noise(0.001, RNGSEED)
        num_params = 100
        num_repetitions = 100
        params = np.ones(num_params)

        sample = [
            diff
            for diff in (params - preprocessor(params))
            for _ in range(num_repetitions)
        ]

        np.testing.assert_allclose(np.std(sample), 0.001, atol=1e-03)

    def test_does_not_mutate_parameters(self):
        preprocessor = add_noise(0.1, RNGSEED)
        params = np.ones(3)

        preprocessor(params)

        np.testing.assert_array_equal(params, np.ones(3))


class TestAbstractMethodsCannotBeCalled:
    # @pytest.mark.parametrize(
    #     "abstract_object",
    #     [CostFunction(), EstimationTasksFactory(), ParameterPreprocessor()],
    # )
    def test_call_abstract_method_raises_errors(self):

        with pytest.raises(TypeError):
            CostFunction()
