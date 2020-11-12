"""Utilities for converting symbolic expressions between different dialects."""
import operator
from functools import singledispatch
from numbers import Number
from typing import NamedTuple, Any, Iterable, Union, Dict, Callable, Tuple

import pyquil
from pyquil import quilatom
import sympy


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


@singledispatch
def translate_expression(
    expression: Union[Expression, Tuple[Expression, ...]],
    dialect: ExpressionDialect
):
    pass


@translate_expression.register
def translate_number(number: Number, dialect: ExpressionDialect):
    return dialect.number_factory(number)


@translate_expression.register
def translate_symbol(symbol: Symbol, dialect: ExpressionDialect):
    return dialect.symbol_factory(symbol)


@translate_expression.register
def translate_function_call(
    function_call: FunctionCall, dialect: ExpressionDialect
):
    if function_call.name not in dialect.known_functions:
        raise ValueError(f"Function {function_call.name} not know in this dialect.")

    return dialect.known_functions[function_call.name](
        *translate_tuple(function_call.args, dialect)
    )


def translate_tuple(expression_tuple: Iterable[Expression], dialect: ExpressionDialect):
    return tuple(
        translate_expression(element, dialect)
        for element in expression_tuple
    )



