"""Save conditions possible to use with recorder."""
from typing import Any

from typing_extensions import Protocol


class SaveCondition(Protocol):
    """Protocol of a function determining if given call should should be saved in the history."""

    def __call__(self, value: Any, params: Any, call_number: int) -> bool:
        """Determine whether current call should be saved in the history.

        Suppose the recorder is constructed for a function `f`, and the params
        `x` are passed to `f` such that `y`=`f(x)`. Then, if this is `n-th`
        evaluation of the function, the value of __call__(y, x, n) determines
        if current call should be saved to the history.

        :param value: current value of the function.
        :param params: parameters passed to the function.
        :param call_number: a natural number determining how many times the target
         function has been called.
        :return: A boolean indicating whether the call being processed should be saved to
        history.
        """
        pass


def always(value: Any, params: Any, call_number: int) -> bool:
    """Default save condition: save always."""
    return True


def every_nth(n: int) -> SaveCondition:
    """Save condition: every n-th step, counting from zero-th one.

    Note: this is factory function, i.e. it returns the actual save condition for given n.
    """

    def _save_condition(value: Any, params: Any, call_number: int) -> bool:
        return call_number % n == 0

    return _save_condition
