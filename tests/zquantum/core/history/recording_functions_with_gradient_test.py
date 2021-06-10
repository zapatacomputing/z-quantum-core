"""Test cases for recording functions with gradients."""
import numpy as np
import pytest
from zquantum.core.history.example_functions import Function2, Function5, function_1
from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import CallableWithGradient


@pytest.mark.parametrize(
    "function,params",
    [
        (function_1, np.array([3, 4])),
        (Function2(5), np.array([-1, 0, 1])),
        (Function5(10), np.array([1, 2, 3])),
    ],
)
def test_recorder_propagates_calls_to_wrapped_functions_and_its_gradient(
    function: CallableWithGradient, params: np.ndarray
):
    target = recorder(function)
    assert target(params) == function(params)
    np.testing.assert_array_almost_equal(
        target.gradient(params), function.gradient(params)
    )


@pytest.mark.parametrize(
    "function,params",
    [
        (function_1, np.array([3, 4])),
        (Function2(5), np.array([-1, 0, 1])),
        (Function5(10), np.array([1, 2, 3])),
    ],
)
def test_calls_to_gradient_are_also_recorded(
    function: CallableWithGradient, params: np.ndarray
):
    target = recorder(function)
    np.testing.assert_array_equal(function.gradient(params), target.gradient(params))

    assert len(target.gradient.history) == 1
    entry = target.gradient.history[0]

    assert entry.call_number == 0
    np.testing.assert_array_equal(entry.params, params)
    np.testing.assert_array_equal(entry.value, target.gradient(params))
