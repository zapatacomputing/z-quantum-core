"""Test cases for recording functions with gradients."""
import pytest
import numpy as np
from .example_functions import function_1, Function2, Function5
from .recorder import recorder
from ..interfaces.functions import CallableWithGradient


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
    assert np.array_equal(target.gradient(params), function.gradient(params))
