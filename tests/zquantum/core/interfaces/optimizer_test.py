import numpy as np
from openfermion import IsingOperator
from scipy.optimize.optimize import OptimizeResult
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.interfaces.mock_objects import MockAnsatz, MockOptimizer
from zquantum.core.interfaces.optimizer import MetaOptimizer


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


class MockMetaOptimizer(MetaOptimizer):
    def minimize(self, parameters):
        etf = self._estimation_tasks_factory(self._ansatz)
        cf = self._cost_function_factory(etf)
        return OptimizeResult(opt_value=cf(parameters))


ansatz = MockAnsatz(number_of_layers=1, problem_size=1)


def simple_estimation_tasks_factory(ansatz: Ansatz):
    return EstimationTask(
        IsingOperator("Z0 Z1", 5), ansatz.parametrized_circuit, number_of_shots=None
    )


def simple_cost_function_factory(estimation_tasks_factory):
    return simple_cost_function


def test_meta_optimizer_initializes_shit_properly_to_bring_up_code_coverage():
    inner_optimizer = MockOptimizer()
    optimizer = MockMetaOptimizer(
        ansatz,
        inner_optimizer,
        simple_estimation_tasks_factory,
        simple_cost_function_factory,
    )
    assert optimizer._ansatz == ansatz
    assert optimizer._inner_optimizer == inner_optimizer
    assert optimizer._estimation_tasks_factory == simple_estimation_tasks_factory
    assert optimizer._cost_function_factory == simple_cost_function_factory
