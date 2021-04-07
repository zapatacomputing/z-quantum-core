from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.interfaces.functions import (
    function_with_gradient,
    has_store_artifact_param,
)


def test_adding_gradient_to_function_storing_artifacts_makes_a_callable_that_stores_artifacts():
    def _test_function(params, store_artifact=None):
        if store_artifact:
            store_artifact("x", params[0])
        return (params ** 2).sum()

    function = function_with_gradient(
        _test_function, finite_differences_gradient(_test_function)
    )
    assert has_store_artifact_param(function)


def test_adding_gradient_to_function_not_storing_artifacts_makes_a_callable_not_storing_artifacts():
    def _test_function(params):
        return (params ** 2).sum()

    function = function_with_gradient(
        _test_function, finite_differences_gradient(_test_function)
    )
    assert not has_store_artifact_param(function)
