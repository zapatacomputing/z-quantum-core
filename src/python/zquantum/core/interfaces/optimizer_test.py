import unittest
import numpy as np
import pytest
from zquantum.core.interfaces.functions import FunctionWithGradient

from .optimizer import optimization_result
from ..gradients import finite_differences_gradient


def rosenbrock_function(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def sum_x_squared(x):
    return sum(x ** 2.0)


class OptimizerTests(object):
    # To run tests with this base class, the following variables need to be properly initialized in the child class:
    # self.optimizers

    def test_optimizer_succeeds_with_optimizing_rosenbrock_function(self):
        for optimizer in self.optimizers:
            cost_function = FunctionWithGradient(rosenbrock_function, finite_differences_gradient(rosenbrock_function))

            results = optimizer.minimize(cost_function, initial_params=np.array([0, 0]))
            self.assertAlmostEqual(results.opt_value, 0, places=4)
            self.assertAlmostEqual(results.opt_params[0], 1, places=3)
            self.assertAlmostEqual(results.opt_params[1], 1, places=3)

            self.assertIn("nfev", results.keys())
            self.assertIn("nit", results.keys())
            self.assertIn("opt_value", results.keys())
            self.assertIn("opt_params", results.keys())
            self.assertIn("history", results.keys())

    def test_optimizer_succeeds_with_optimizing_sum_of_squares_function(self):
        for optimizer in self.optimizers:
            cost_function = FunctionWithGradient(sum_x_squared, finite_differences_gradient(sum_x_squared))

            results = optimizer.minimize(
                cost_function, initial_params=np.array([1, -1])
            )
            self.assertAlmostEqual(results.opt_value, 0, places=5)
            self.assertAlmostEqual(results.opt_params[0], 0, places=4)
            self.assertAlmostEqual(results.opt_params[1], 0, places=4)

            self.assertIn("nfev", results.keys())
            self.assertIn("nit", results.keys())
            self.assertIn("opt_value", results.keys())
            self.assertIn("opt_params", results.keys())
            self.assertIn("history", results.keys())


def test_optimization_result_contains_opt_value_and_opt_params():
    opt_value = 2.0
    opt_params = [-1, 0, 3.2]

    result = optimization_result(opt_value=opt_value, opt_params=opt_params)

    assert result.opt_value == opt_value
    assert result.opt_params == opt_params


def test_optimization_result_contains_other_attributes_passed_as_kwargs():
    opt_value = 0.0
    opt_params = [1, 2, 3]
    kwargs = {"bitstring": "01010", "foo": 3.0}

    result = optimization_result(opt_value=opt_value, opt_params=opt_params, **kwargs)

    assert all(getattr(result, key) == value for key, value in kwargs.items())
