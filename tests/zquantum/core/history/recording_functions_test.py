"""Test cases for recording basic functions."""
from unittest.mock import Mock, call

import numpy as np
import pytest
from zquantum.core.history.example_functions import (
    Function2,
    function_1,
    sum_of_squares,
)
from zquantum.core.history.recorder import recorder
from zquantum.core.history.save_conditions import SaveCondition, every_nth


@pytest.mark.parametrize(
    "source_function,param",
    [
        (np.sin, 10),
        (sum_of_squares, [1, 2, 3]),
        (function_1, np.array([2, 3])),
        (Function2(5), np.array([1, 2, 3])),
    ],
)
def test_recorder_propagates_calls_to_wrapped_function(source_function, param):
    function = recorder(source_function)
    assert function(param) == source_function(param)


@pytest.mark.parametrize(
    "source_function,params_sequence",
    [
        (np.exp, [1, 3, 5, 10]),
        (sum_of_squares, [[1, 2, 3], [4, 5, 6]]),
        (function_1, [np.array([-2, -3]), np.array([0, 1])]),
        (Function2(5), [np.array([1, 0, -1]), np.array([0, 1, 2])]),
    ],
)
def test_recorder_does_not_make_any_redundant_calls_to_wrapped_function(
    source_function, params_sequence
):
    spy = Mock(wraps=source_function)
    function = recorder(spy)

    for params in params_sequence:
        function(params)

    assert spy.call_args_list == [call(params) for params in params_sequence]


@pytest.mark.parametrize(
    "source_function,params_sequence",
    [
        (np.exp, [1, 3, 5, 10]),
        (sum_of_squares, [[1, 2, 3], [4, 5, 6]]),
        (function_1, [np.array([-2, -3]), np.array([0, 1]), np.array([1, 2])]),
        (
            Function2(5),
            [
                np.array(args)
                for args in [[-1, 0, 1], [10, 20, 30], [0, 1, 2], [3, 4, 5]]
            ],
        ),
    ],
)
def test_by_default_recorder_records_all_evaluations(source_function, params_sequence):
    function = recorder(source_function)

    for params in params_sequence:
        function(params)

    assert [entry.params for entry in function.history] == params_sequence
    assert [entry.value for entry in function.history] == [
        source_function(params) for params in params_sequence
    ]
    assert [entry.call_number for entry in function.history] == list(
        range(len(params_sequence))
    )


@pytest.mark.parametrize(
    "source_function,params_sequence,condition",
    [
        (sum_of_squares, [list(range(n)) for n in range(100)], every_nth(5)),
        (function_1, [np.array([k, k + 1]) for k in range(1000)], every_nth(101)),
        (
            Function2(10),
            [np.array([k, 2 * k, 3 * k]) for k in range(200)],
            every_nth(21),
        ),
    ],
)
def test_recorder_records_only_calls_for_which_save_condition_evaluates_to_true(
    source_function, params_sequence, condition: SaveCondition
):
    function = recorder(source_function, save_condition=condition)
    expected_values, expected_params, expected__call_numbers = zip(
        *filter(
            lambda x: condition(*x),
            (
                (source_function(params), params, i)
                for i, params in enumerate(params_sequence)
            ),
        )
    )
    for params in params_sequence:
        function(params)

    assert [entry.call_number for entry in function.history] == list(
        expected__call_numbers
    )
    assert [entry.params for entry in function.history] == list(expected_params)
    assert [entry.value for entry in function.history] == list(expected_values)
