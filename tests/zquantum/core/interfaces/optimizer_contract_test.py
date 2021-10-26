from typing import Callable

import numpy as np
import pytest
from zquantum.core.interfaces.cost_function import CostFunction
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


@pytest.mark.parametrize("contract", NESTED_OPTIMIZER_CONTRACTS)
def test_validate_contracts(contract):
    assert contract(
        _good_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )
    assert not contract(
        _malicious_nested_optimizer,
        mock_cost_function_factory,
        np.array([2]),
    )
