from typing import Callable, List

import numpy as np
from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.functions import FunctionWithGradient
from zquantum.core.interfaces.mock_objects import (
    MockNestedOptimizer,
    MockOptimizer,
    mock_cost_function,
)
from zquantum.core.interfaces.optimizer_test import NESTED_OPTIMIZER_CONTRACTS


class MaliciousNestedOptimizer(MockNestedOptimizer):
    def _minimize(
        self,
        cost_function_factory: Callable[[int], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ):
        keep_history = not keep_history
        results = super()._minimize(
            cost_function_factory, initial_params, keep_history=keep_history
        )
        del results["nit"]
        return results


_good_nested_optimizer = MockNestedOptimizer(inner_optimizer=MockOptimizer(), n_iters=5)
_malicious_nested_optimizer = MaliciousNestedOptimizer(
    inner_optimizer=MockOptimizer(), n_iters=5
)


def mock_cost_function_factory(iteration_id: int):
    def modified_cost_function(params):
        return mock_cost_function(params) ** iteration_id

    return modified_cost_function


def mock_cost_function_factory_with_gradient(iteration_id: int):
    cost_function = mock_cost_function_factory(iteration_id)

    return FunctionWithGradient(
        cost_function, finite_differences_gradient(cost_function)
    )


def test_validate_nested_optimizer_records_history_if_keep_history_is_true():
    assert NESTED_OPTIMIZER_CONTRACTS[0](
        _good_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )
    assert not NESTED_OPTIMIZER_CONTRACTS[0](
        _malicious_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )


def test_validate_nested_optimizer_records_gradient_history_if_keep_history_is_true():
    assert NESTED_OPTIMIZER_CONTRACTS[1](
        _good_nested_optimizer,
        mock_cost_function_factory_with_gradient,
        np.array([2]),
    )
    assert not NESTED_OPTIMIZER_CONTRACTS[1](
        _malicious_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )


def test_validate_nested_optimizer_does_not_record_history_if_keep_history_is_false():
    assert NESTED_OPTIMIZER_CONTRACTS[2](
        _good_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )
    assert not NESTED_OPTIMIZER_CONTRACTS[2](
        _malicious_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )


def test_validate_nested_optimizer_does_not_record_history_by_default():
    assert NESTED_OPTIMIZER_CONTRACTS[3](
        _good_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )
    assert not NESTED_OPTIMIZER_CONTRACTS[3](
        _malicious_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )


def test_validate_nested_optimizer_returns_all_the_mandatory_fields_in_results():
    assert NESTED_OPTIMIZER_CONTRACTS[4](
        _good_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )
    assert not NESTED_OPTIMIZER_CONTRACTS[4](
        _malicious_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )
