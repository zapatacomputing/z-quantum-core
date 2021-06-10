from copy import copy, deepcopy

import pytest
from zquantum.core.history.example_functions import (
    Function2,
    Function5,
    function_1,
    function_3,
    function_4,
    function_6,
)
from zquantum.core.history.recorder import recorder


@pytest.mark.parametrize("func", [Function2(5), function_3, function_4, Function5(0.5)])
def test_recorder_can_be_copied_shallowly(func):
    recorded = recorder(func)

    recorded_copy = copy(recorded)

    assert recorded_copy.target is recorded.target
    assert recorded_copy.predicate is recorded.predicate


@pytest.mark.parametrize("func", [Function2(5), function_3, function_4, Function5(0.5)])
def test_recorder_can_be_copied_deeply(func):
    recorded = recorder(func)

    recorded_copy = deepcopy(recorded)

    assert recorded_copy.target == recorded.target
    assert recorded_copy.predicate == recorded.predicate
