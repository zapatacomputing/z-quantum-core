"""Test cases for save conditions."""
import pytest

from .save_conditions import always, every_nth


@pytest.mark.parametrize(
    "value,params,call_number", [(10, [1, 2, 3], 0), (5.0, [1, 2, 3], 1), (0.5, 3.0, 2)]
)
def test_always_returns_true_independently_from_its_arguments(
    value, params, call_number
):
    assert always(value, params, call_number)


@pytest.mark.parametrize(
    "n,value,param,call_number",
    [
        (5, 3.1, (1, 2, 3), 10),
        (5, [4, 2], (0, 0.1, 0.2, 0.3), 1000),
        (5, "test", (0.0, 0.5), 1),
        (5, (1, 2, 3, 4), (1, 2, 3), 2001),
        (1000, ["foo", "bar"], (4, 5, 6), 10),
        (1000, [-1, 0, 1], (10, 20, 30), 2000),
        (100, (2.0, 3.0), (0, 1, 2), 0),
    ],
)
def test_every_nth_returns_true_for_call_numbers_divisible_by_n_and_false_otherwise(
    n, value, param, call_number
):
    assert every_nth(n)(value, param, call_number) == (call_number % n == 0)
