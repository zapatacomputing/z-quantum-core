"""Utilities for converting sympy expressions to our native Expression format."""
import operator
from functools import singledispatch
from numbers import Number

import sympy

from .expressions import ExpressionDialect, FunctionCall, Symbol


def is_multiplication_by_reciprocal(sympy_mul: sympy.Mul) -> bool:
    """Check if given sympy multiplication is of the form x * (1 / y)."""
    args = sympy_mul.args
    return len(args) == 2 and isinstance(args[1], sympy.Pow) and args[1].args[1] == -1


def is_addition_of_negation(sympy_add: sympy.Add) -> bool:
    """Check if given sympy addition is of the form x + (-y)."""
    args = sympy_add.args
    return len(args) == 2 and isinstance(args[1], sympy.Mul) and args[1].args[0] == -1


@singledispatch
def expression_from_sympy(expression):
    """Parse Sympy expression into intermediate expression tree."""
    raise NotImplementedError(
        f"Expression {expression} of type {type(expression)} is currently not supported"
    )


@expression_from_sympy.register
def identity(number: Number):
    return number


@expression_from_sympy.register
def symbol_from_sympy(symbol: sympy.Symbol):
    return Symbol(str(symbol))


@expression_from_sympy.register
def native_integer_from_sympy_integer(number: sympy.Integer):
    return int(number)


@expression_from_sympy.register
def native_float_from_sympy_float(number: sympy.Float):
    return float(number)


@expression_from_sympy.register
def native_float_from_sympy_rational(number: sympy.Rational):
    return float(number)


@expression_from_sympy.register
def native_imaginary_unit_from_sympy_imaginary_unit(
    _unit: sympy.core.numbers.ImaginaryUnit,
):
    return 1j


def _negate_sympy_expr(expr):
    return expr * (-1)


@expression_from_sympy.register
def addition_from_sympy_add(add: sympy.Add):
    if is_addition_of_negation(add):
        return FunctionCall(
            "sub",
            (
                expression_from_sympy(add.args[0]),
                expression_from_sympy(_negate_sympy_expr(add.args[1])),
            ),
        )
    return FunctionCall("add", expression_from_sympy(add.args))


@expression_from_sympy.register
def multiplication_from_sympy_mul(mul: sympy.Mul):
    if is_multiplication_by_reciprocal(mul):
        return FunctionCall(
            "div",
            (
                expression_from_sympy(mul.args[0]),
                expression_from_sympy(mul.args[1].args[0]),
            ),
        )
    else:
        return FunctionCall("mul", expression_from_sympy(mul.args))


@expression_from_sympy.register
def power_from_sympy_pow(power: sympy.Pow):
    if power.args[1] == -1:
        return FunctionCall("div", (1, expression_from_sympy(power.args[0])))
    elif power.args[1] == 0.5:
        return FunctionCall("sqrt", (expression_from_sympy(power.args[0]),))
    else:
        return FunctionCall("pow", expression_from_sympy(power.args))


@expression_from_sympy.register
def function_call_from_sympy_function(function: sympy.Function):
    return FunctionCall(str(function.func), expression_from_sympy(function.args))


@expression_from_sympy.register
def expression_tuple_from_tuple_of_sympy_args(args: tuple):
    return tuple(expression_from_sympy(arg) for arg in args)


# Dialect defining conversion of intermediate expression tree to
# the expression based on Sympy functions/Symbols
# This is intended to be passed by a `dialect` argument of `translate_expression`.
SYMPY_DIALECT = ExpressionDialect(
    symbol_factory=lambda symbol: sympy.Symbol(symbol.name),
    number_factory=lambda number: number,
    known_functions={
        "add": operator.add,
        "mul": operator.mul,
        "div": operator.truediv,
        "sub": operator.sub,
        "pow": operator.pow,
        "cos": sympy.cos,
        "sin": sympy.sin,
        "exp": sympy.exp,
        "sqrt": sympy.sqrt,
        "tan": sympy.tan,
    },
)
