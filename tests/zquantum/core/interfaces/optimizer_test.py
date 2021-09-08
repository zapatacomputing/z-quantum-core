import numpy as np
from zquantum.core.interfaces.mock_objects import MockOptimizer


class CustomMockOptimizer(MockOptimizer):
    def _preprocess_cost_function(self, cost_function):
        def _new_cost_function(parameters):
            return cost_function(parameters) + 1

        return _new_cost_function


def simple_cost_function(parameters):
    return 1


def test_cost_function_is_preprocessed_before_minimization():
    optimizer = CustomMockOptimizer()
    result = optimizer.minimize(simple_cost_function, np.zeros(4))

    assert result.opt_value == 2
