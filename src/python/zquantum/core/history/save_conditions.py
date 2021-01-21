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

        Args:
            value: current value of the function.
            params: parameters passed to the function.
            call_number: a natural number determining how many times the target
              function has been called.

        Returns:
            A boolean indicating whether the call being processed should be
            saved to history.
        """
        pass


def always(value: Any, params: Any, call_number: int) -> bool:
    """Default save condition: save always.

    See parameters in SaveCondition.__call__ for explanation."""
    return True


def every_nth(n: int) -> SaveCondition:
    """Save condition: every n-th step, counting from zero-th one.

    Note: this is factory function, i.e. it returns the actual save condition
    for given n.

    Args:
        n: the integer determining which steps should be saved in history.

    Returns:
        Function `f` implementing the SaveCondition protocol, such that
        f(y, x, k) is True if and only if k = 0 (mod n).
    """

    def _save_condition(value: Any, params: Any, call_number: int) -> bool:
        return call_number % n == 0

    return _save_condition
