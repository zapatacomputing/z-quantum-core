"""Test case prototypes that can be used in other projects.

Note that this file won't be executed on its own by pytest.
You need to define your own test cases that inherit from the ones defined here.
"""


import numpy as np
import pytest
from zquantum.core.interfaces.functions import FunctionWithGradient

from ..gradients import finite_differences_gradient
from ..history.recorder import recorder

MANDATORY_OPTIMIZATION_RESULT_FIELDS = ("nfev", "nit", "opt_value", "opt_params")


class OptimizerTests(object):
    """Base class for optimizers tests.

    How to use:
    1. Inherit this class (remember to start name of the class with "Test"
    2. In the same module define fixture called "optimizer".

    Basic usage pattern:

    @pytest.fixture
    def optimizer():
        return MyOptimizer()


    class TestMyOptimizer(OptimizerTests): # Inherits all tests from this class
         def test_some_new_feature(self, optimizer): # new test
             ....

    Notice that the `optimizer` fixture can be parametrized if you wish to
    perform tests for various configurations of your optimizer.
    """

    @pytest.fixture
    def sum_x_squared(self):
        return lambda x: sum(x ** 2.0)

    @pytest.fixture
    def rosenbrock_function(self):
        """The Rosenbrock function"""
        return lambda x: sum(
            100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0
        )

    def test_optimizer_succeeds_with_optimizing_rosenbrock_function(
        self, optimizer, rosenbrock_function, keep_history
    ):
        cost_function = FunctionWithGradient(
            rosenbrock_function, finite_differences_gradient(rosenbrock_function)
        )

        results = optimizer.minimize(
            cost_function, initial_params=np.array([0, 0]), keep_history=keep_history
        )
        assert results.opt_value == pytest.approx(0, abs=1e-4)
        assert results.opt_params == pytest.approx(np.ones(2), abs=1e-3)

        assert all(field in results for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)

        assert "history" in results or not keep_history
        assert "gradient_history" in results or not keep_history

    def test_optimizer_succeeds_with_optimizing_sum_of_squares_function(
        self, optimizer, sum_x_squared, keep_history
    ):

        cost_function = FunctionWithGradient(
            sum_x_squared, finite_differences_gradient(sum_x_squared)
        )

        results = optimizer.minimize(
            cost_function, initial_params=np.array([1, -1]), keep_history=keep_history
        )

        assert results.opt_value == pytest.approx(0, abs=1e-5)
        assert results.opt_params == pytest.approx(np.zeros(2), abs=1e-4)

        assert all(field in results for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)

        assert "history" in results or not keep_history
        assert "gradient_history" in results or not keep_history

    def test_optimizer_succeeds_on_cost_function_without_gradient(
        self, optimizer, sum_x_squared, keep_history
    ):
        cost_function = sum_x_squared

        results = optimizer.minimize(
            cost_function, initial_params=np.array([1, -1]), keep_history=keep_history
        )
        assert results.opt_value == pytest.approx(0, abs=1e-5)
        assert results.opt_params == pytest.approx(np.zeros(2), abs=1e-4)

        assert all(field in results for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)

        assert "history" in results or not keep_history
        assert "gradient_history" not in results

    def test_optimizer_records_history_if_keep_history_is_true(
        self, optimizer, sum_x_squared
    ):

        # To check that history is recorded correctly, we wrap cost_function
        # with a recorder. Optimizer should wrap it a second time and
        # therefore we can compare two histories to see if they agree.
        cost_function = recorder(sum_x_squared)

        result = optimizer.minimize(cost_function, np.array([-1, 1]), keep_history=True)

        for result_history_entry, cost_function_history_entry in zip(
            result.history, cost_function.history
        ):
            assert (
                result_history_entry.call_number
                == cost_function_history_entry.call_number
            )
            assert np.allclose(
                result_history_entry.params, cost_function_history_entry.params
            )
            assert np.isclose(
                result_history_entry.value, cost_function_history_entry.value
            )

    def test_gradients_history_is_recorded_if_keep_history_is_true(
        self, optimizer, sum_x_squared
    ):
        # To check that history is recorded correctly, we wrap cost_function
        # with a recorder. Optimizer should wrap it a second time and
        # therefore we can compare two histories to see if they agree.
        cost_function = recorder(
            FunctionWithGradient(
                sum_x_squared, finite_differences_gradient(sum_x_squared)
            )
        )

        result = optimizer.minimize(cost_function, np.array([-1, 1]), keep_history=True)
        assert len(result.gradient_history) == len(cost_function.gradient.history)

        for result_history_entry, cost_function_history_entry in zip(
            result.gradient_history, cost_function.gradient.history
        ):
            assert (
                result_history_entry.call_number
                == cost_function_history_entry.call_number
            )
            assert np.allclose(
                result_history_entry.params, cost_function_history_entry.params
            )
            assert np.allclose(
                result_history_entry.value, cost_function_history_entry.value
            )

    def test_optimizer_does_not_record_history_if_keep_history_is_set_to_false(
        self, optimizer, sum_x_squared
    ):
        result = optimizer.minimize(
            sum_x_squared, np.array([-2, 0.5]), keep_history=False
        )

        assert result.history == []

    def test_optimizer_does_not_record_history_if_keep_history_by_default(
        self, optimizer, sum_x_squared
    ):
        result = optimizer.minimize(sum_x_squared, np.array([-2, 0.5]))

        assert result.history == []
