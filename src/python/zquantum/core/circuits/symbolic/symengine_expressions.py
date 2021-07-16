"""Utilities for converting symengine expressions to our native Expression format."""
import operator
from functools import singledispatch
from numbers import Number

import symengine
from symengine.lib.symengine_wrapper import ImaginaryUnit

from .expressions import ExpressionDialect, FunctionCall, Symbol, reduction


def is_multiplication_by_reciprocal(symengine_mul: symengine.Mul) -> bool:
    """Check if given symengine multiplication is of the form x * (1 / y)."""
    args = symengine_mul.args
    return (
        len(args) == 2 and isinstance(args[1], symengine.Pow) and args[1].args[1] == -1
    )


def is_right_addition_of_negation(symengine_add: symengine.Add) -> bool:
    """Check if given symengine addition is of the form x + (-y)."""
    args = symengine_add.args
    return (
        len(args) == 2 and isinstance(args[1], symengine.Mul) and args[1].args[0] == -1
    )


def is_left_addition_of_negation(symengine_add: symengine.Add) -> bool:
    """Check if given symengine addition is of the form (-x) + y."""
    args = symengine_add.args
    return (
        len(args) == 2 and isinstance(args[0], symengine.Mul) and args[0].args[0] == -1
    )


@singledispatch
def expression_from_symengine(expression):
    """Parse symengine expression into intermediate expression tree."""
    raise NotImplementedError(
        f"Expression {expression} of type {type(expression)} is currently not supported"
    )


@expression_from_symengine.register
def identity(number: Number):
    return number


@expression_from_symengine.register
def symbol_from_symengine(symbol: symengine.Symbol):
    return Symbol(str(symbol))


@expression_from_symengine.register
def native_integer_from_symengine_integer(number: symengine.Integer):
    return int(number)


@expression_from_symengine.register
def native_float_from_symengine_float(number: symengine.Float):
    return float(number)


@expression_from_symengine.register
def native_float_from_symengine_rational(number: symengine.Rational):
    return float(number)


@expression_from_symengine.register
def native_imaginary_unit_from_symengine_imaginary_unit(
    _unit: ImaginaryUnit,
):
    return 1j


def _negate_symengine_expr(expr):
    return expr * (-1)


@expression_from_symengine.register
def addition_from_symengine_add(add: symengine.Add):
    if is_left_addition_of_negation(add):
        return FunctionCall(
            "sub",
            (
                expression_from_symengine(add.args[1]),
                expression_from_symengine(_negate_symengine_expr(add.args[0])),
            ),
        )
    elif is_right_addition_of_negation(add):
        return FunctionCall(
            "sub",
            (
                expression_from_symengine(add.args[0]),
                expression_from_symengine(_negate_symengine_expr(add.args[1])),
            ),
        )
    return FunctionCall("add", expression_from_symengine(add.args))


@expression_from_symengine.register
def multiplication_from_symengine_mul(mul: symengine.Mul):
    if is_multiplication_by_reciprocal(mul):
        return FunctionCall(
            "div",
            (
                expression_from_symengine(mul.args[0]),
                expression_from_symengine(mul.args[1].args[0]),
            ),
        )
    else:
        return FunctionCall("mul", expression_from_symengine(mul.args))


@expression_from_symengine.register
def power_from_symengine_pow(power: symengine.Pow):
    if power.args[1] == -1:
        return FunctionCall("div", (1, expression_from_symengine(power.args[0])))
    elif power.args[1] == 0.5:
        return FunctionCall("sqrt", (expression_from_symengine(power.args[0]),))
    elif power.args[0] == symengine.E:
        return FunctionCall("exp", (expression_from_symengine(power.args[1]),))
    else:
        return FunctionCall("pow", expression_from_symengine(power.args))


@expression_from_symengine.register
def function_call_from_symengine_function(function: symengine.Function):
    return FunctionCall(
        str(type(function).__name__), expression_from_symengine(function.args)
    )


@expression_from_symengine.register
def expression_tuple_from_tuple_of_symengine_args(args: tuple):
    return tuple(expression_from_symengine(arg) for arg in args)


# Dialect defining conversion of intermediate expression tree to
# the expression based on symengine functions/Symbols
# This is intended to be passed by a `dialect` argument of `translate_expression`.
SYMENGINE_DIALECT = ExpressionDialect(
    symbol_factory=lambda symbol: symengine.Symbol(symbol.name),
    number_factory=lambda number: number,
    known_functions={
        "add": reduction(operator.add),
        "mul": reduction(operator.mul),
        "div": operator.truediv,
        "sub": operator.sub,
        "pow": operator.pow,
        "cos": symengine.cos,
        "sin": symengine.sin,
        "exp": symengine.exp,
        "sqrt": symengine.sqrt,
        "tan": symengine.tan,
    },
)
