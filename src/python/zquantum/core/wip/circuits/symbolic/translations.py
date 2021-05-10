"""Utilities related to translation of symbolic expressions."""
from functools import singledispatch
from numbers import Number
from typing import Iterable, Tuple, Union

from .expressions import Expression, ExpressionDialect, FunctionCall, Symbol


@singledispatch
def translate_expression(
    expression: Union[Expression, Tuple[Expression, ...]], dialect: ExpressionDialect
):
    pass


@translate_expression.register
def translate_number(number: Number, dialect: ExpressionDialect):
    return dialect.number_factory(number)


@translate_expression.register
def translate_symbol(symbol: Symbol, dialect: ExpressionDialect):
    return dialect.symbol_factory(symbol)


@translate_expression.register
def translate_function_call(function_call: FunctionCall, dialect: ExpressionDialect):
    if function_call.name not in dialect.known_functions:
        raise ValueError(f"Function {function_call.name} is unknown in this dialect.")

    return dialect.known_functions[function_call.name](
        *translate_tuple(function_call.args, dialect)
    )


def translate_tuple(expression_tuple: Iterable[Expression], dialect: ExpressionDialect):
    return tuple(translate_expression(element, dialect) for element in expression_tuple)
