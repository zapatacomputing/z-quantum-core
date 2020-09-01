"""Save conditions possible to use with recorder."""
from typing import Any

from typing_extensions import Protocol


class SaveCondition(Protocol):
    def __call__(self, value: Any, params: Any, call_number: int) -> bool:
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
