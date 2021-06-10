import pytest
from zquantum.core.history.example_functions import (
    Function2,
    Function5,
    function_3,
    function_4,
)
from zquantum.core.history.recorder import recorder


@pytest.mark.parametrize("func", [Function2(5), function_3, function_4, Function5(0.5)])
def test_recorder_correctly_gets_attributes_of_recorder_function(func):
    func.custom_attribute = "custom-attr"

    recorded = recorder(func)

    assert recorded.custom_attribute == func.custom_attribute

    func.custom_attribute = "some-other-value"

    assert recorded.custom_attribute == func.custom_attribute


@pytest.mark.parametrize("func", [Function2(5), function_3, function_4, Function5(0.5)])
def test_recorder_correctly_sets_attributes_of_recorder_function(func):
    func.custom_attribute = "custom-attr"

    recorded = recorder(func)

    recorded.custom_attribute = "new-value"

    assert func.custom_attribute == "new-value"
