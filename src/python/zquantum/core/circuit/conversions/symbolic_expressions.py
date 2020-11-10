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


def is_multiplication_by_reciprocal(sympy_mul: sympy.Mul) -> bool:
    """Check if given sympy multiplication is of the form x * (1 / y)."""
    args = sympy_mul.args
    return len(args) == 2 and isinstance(args[1], sympy.Pow) and args[1].args[1] == -1


def is_addition_of_negation(sympy_add: sympy.Add) -> bool:
    """Check if given sympy addition os of the form x + (-y)."""
    args = sympy_add.args
    return len(args) == 2 and isinstance(args[1], sympy.Mul) and args[1].args[0] == -1


@singledispatch
def expression_from_sympy(expression):
    raise NotImplementedError(
        f"Expression {expression} of type {type(expression)} is currentlyl not supported"
    )


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
def native_imaginary_unit_from_sympy_imaginary_unit(_unit: sympy.numbers.ImaginaryUnit):
    return 1j


@expression_from_sympy.register
def addition_from_sympy_add(add: sympy.Add):
    if is_addition_of_negation(add):
        return FunctionCall(
            "sub",
            (
                expression_from_sympy(add.args[0]),
                expression_from_sympy(add.args[1].args[1]),
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
    return FunctionCall("pow", expression_from_sympy(power.args))


@expression_from_sympy.register
def function_call_from_sympy_function(function: sympy.Function):
    return FunctionCall(str(function.func), expression_from_sympy(function.args))


@expression_from_sympy.register
def expression_tuple_from_tuple_of_sympy_args(args: tuple):
    return tuple(expression_from_sympy(arg) for arg in args)


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


QUIL_DIALECT = ExpressionDialect(
    symbol_factory=lambda symbol: pyquil.quil.Parameter(symbol.name),
    number_factory=lambda number: number,
    known_functions={
        "add": operator.add,
        "mul": operator.mul,
        "div": operator.truediv,
        "sub": operator.sub,
        "cos": quilatom.quil_cos,
        "sin": quilatom.quil_sin,
        "exp": quilatom.quil_exp,
        "sqrt": quilatom.quil_sqrt
    }
)
