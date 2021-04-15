"""Utilities for converting symbolic expressions between different dialects."""
from functools import reduce
from numbers import Number
from typing import Any, Callable, Dict, Iterable, NamedTuple, Union


class Symbol(NamedTuple):
    """Abstract symbol."""

    name: str


class FunctionCall(NamedTuple):
    """Represents abstract function call.     """

    name: str
    args: Iterable["Expression"]


Expression = Union[Symbol, FunctionCall, Number]


class ExpressionDialect(NamedTuple):
    """Dialect of arithmetic expression.

    This is to group information on how to transform expression given in
    our native representation into some representation in external
    library (e.g. PyQuil or Sympy).
    """

    symbol_factory: Callable[[Symbol], Any]
    number_factory: Callable[[Number], Any]
    known_functions: Dict[str, Callable[..., Any]]


def reduction(operator):
    def _reduction(*args):
        return reduce(operator, args)

    return _reduction
