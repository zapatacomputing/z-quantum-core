"""Test cases for sympy_expressions module."""
import pytest
import sympy
from .expressions import Symbol, FunctionCall
from .sympy_expressions import (
    expression_from_sympy,
    is_addition_of_negation,
    is_multiplication_by_reciprocal,
)


@pytest.mark.parametrize("number", [3, 4.0, 1j, 3.0 - 2j])
def test_native_numbers_are_preserved(number):
    assert expression_from_sympy(number) == number


@pytest.mark.parametrize(
    "sympy_symbol, expected_symbol",
    [
        (sympy.Symbol("theta"), Symbol("theta")),
        (sympy.Symbol("x"), Symbol("x")),
        (sympy.Symbol("c_i"), Symbol("c_i")),
    ],
)
def test_symbols_are_converted_to_instance_of_symbol_class(
    sympy_symbol, expected_symbol
):
    assert expression_from_sympy(sympy_symbol) == expected_symbol


@pytest.mark.parametrize(
    "sympy_number, expected_number, expected_class",
    [
        (sympy.sympify(2), 2, int),
        (sympy.sympify(-2.5), -2.5, float),
        (sympy.Rational(3, 8), 0.375, float),
    ],
)
def test_sympy_numbers_are_converted_to_corresponding_native_number(
    sympy_number, expected_number, expected_class
):
    native_number = expression_from_sympy(sympy_number)
    assert native_number == expected_number
    assert isinstance(native_number, expected_class)


def test_imaginary_unit_is_converted_to_1j():
    assert expression_from_sympy(sympy.I) == 1j


# In below functions we explicitly construct Add and Mul objects
# because arithmetic operations on sympy expressions may perform
# additional evaluation which may circumvent our expectations.
@pytest.mark.parametrize(
    "sympy_add, expected_args",
    [
        (sympy.Add(1, 2, 3, evaluate=False), (1, 2, 3)),
        (sympy.Add(sympy.Symbol("x"), 1, evaluate=False), (Symbol("x"), 1)),
        (
            sympy.Add(
                sympy.Symbol("x"),
                sympy.Symbol("y"),
                sympy.Symbol("z"),
                evaluate=False,
            ),
            (Symbol("x"), Symbol("y"), Symbol("z")),
        ),
    ],
)
def test_sympy_add_is_converted_to_function_call_with_add_operation(
    sympy_add, expected_args
):
    assert expression_from_sympy(sympy_add) == FunctionCall("add", expected_args)


@pytest.mark.parametrize(
    "sympy_mul, expected_args",
    [
        (sympy.Mul(4, 2, 3, evaluate=False), (4, 2, 3)),
        (sympy.Mul(sympy.Symbol("x"), 2, evaluate=False), (Symbol("x"), 2)),
        (
            sympy.Mul(
                sympy.Symbol("x"),
                sympy.Symbol("y"),
                sympy.Symbol("z"),
                evaluate=False,
            ),
            (Symbol("x"), Symbol("y"), Symbol("z")),
        ),
    ],
)
def test_sympy_mul_is_converted_to_function_call_with_mul_operation(
    sympy_mul, expected_args
):
    assert expression_from_sympy(sympy_mul) == FunctionCall("mul", expected_args)


@pytest.mark.parametrize(
    "sympy_multiplication",
    [
        sympy.Symbol("x") / sympy.Symbol("y"),
        sympy.Symbol("x") / (sympy.Symbol("z") + 1),
    ],
)
def test_mul_resulting_from_division_is_classified_as_multiplication_by_reciprocal(
    sympy_multiplication,
):
    assert is_multiplication_by_reciprocal(sympy_multiplication)


@pytest.mark.parametrize(
    "sympy_multiplication",
    [
        sympy.Symbol("x") * sympy.Symbol("y"),
        2 * sympy.Symbol("theta"),
        sympy.Symbol("x") * sympy.Symbol("y") * sympy.Symbol("z"),
    ],
)
def test_mul_not_resulting_from_division_is_not_classified_as_multiplication_by_reciprocal(
    sympy_multiplication,
):
    # Note: obviously you can manually construct multiplication that would
    # be classified as multiplication by reciprocal. The bottom line of this
    # test is: usual, simple multiplications are multiplications, not divisions.
    assert not is_multiplication_by_reciprocal(sympy_multiplication)


@pytest.mark.parametrize(
    "sympy_multiplication, expected_args",
    [
        (sympy.Symbol("x") / sympy.Symbol("y"), (Symbol("x"), Symbol("y"))),
        (
            sympy.Symbol("x") / (sympy.Add(sympy.Symbol("z"), 1, evaluate=False)),
            (Symbol("x"), FunctionCall("add", (Symbol("z"), 1))),
        ),
    ],
)
def test_division_is_converted_into_div_function_call_instead_of_multiplication_by_reciprocal(
    sympy_multiplication, expected_args
):
    # Important note about sympy: there is no Div operator (as opposed to
    # e.g. Mul). The division on sympy expressions actually produces Mul
    # objects, in which second operand is a reciprocal of the original one.
    # We need to deal with this case, otherwise converting anything that
    # contains division will result in very confusing expressions.
    assert expression_from_sympy(sympy_multiplication) == FunctionCall(
        "div", expected_args
    )


@pytest.mark.parametrize(
    "sympy_addition",
    [
        sympy.Symbol("x") - sympy.Symbol("y"),
        sympy.Symbol("x") - 1 / sympy.Symbol("y"),
        1 - sympy.Symbol("x")
        # Note: negation of previous case would fail, since in that case
        # sympy would make Add(-1, Symbol("x")) out of it, unlike in other
        # cases where it produces e.g.
        # Add(Symbol("x", Mul(Symbol("y"), -1))).
    ],
)
def test_add_resulting_from_subtraction_is_classified_as_addition_of_negation(
    sympy_addition,
):
    assert is_addition_of_negation(sympy_addition)


@pytest.mark.parametrize(
    "sympy_addition",
    [sympy.Symbol("x") + sympy.Symbol("y"), sympy.Symbol("x") + 10],
)
def test_add_not_resulting_from_subtraction_is_not_classified_as_addition_of_negation(
    sympy_addition,
):
    assert not is_addition_of_negation(sympy_addition)


@pytest.mark.parametrize(
    "sympy_addition, expected_args",
    [
        (sympy.Symbol("x") - sympy.Symbol("y"), (Symbol("x"), Symbol("y"))),
        (1 - sympy.Symbol("x"), (1, Symbol("x"))),
        (
            1 - sympy.Symbol("x") * sympy.Symbol("y"),
            (1, FunctionCall("mul", (Symbol("x"), Symbol("y")))),
        ),
    ],
)
def test_add_resulting_from_subtraction_is_converted_to_sub_function_call(
    sympy_addition, expected_args
):
    assert expression_from_sympy(sympy_addition) == FunctionCall("sub", expected_args)


@pytest.mark.parametrize(
    "sympy_power, expected_args",
    [
        (sympy.Pow(sympy.Symbol("x"), 2), (Symbol("x"), 2)),
        (sympy.Pow(2, sympy.Symbol("x")), (2, Symbol("x"))),
        (
            sympy.Pow(sympy.Symbol("x"), sympy.Symbol("y")),
            (Symbol("x"), Symbol("y")),
        ),
    ],
)
def test_sympy_pow_is_converted_to_pow_function_call(sympy_power, expected_args):
    assert expression_from_sympy(sympy_power) == FunctionCall("pow", expected_args)


@pytest.mark.parametrize(
    "sympy_power, expected_denominator",
    [
        (sympy.Pow(sympy.Symbol("x"), -1), Symbol("x")),
        (
            sympy.Pow(
                sympy.Add(sympy.Symbol("x"), sympy.Symbol("y"), evaluate=False), -1
            ),
            FunctionCall("add", (Symbol("x"), Symbol("y"))),
        ),
    ],
)
def test_sympy_power_with_negative_one_exponent_gets_converted_to_division(
    sympy_power, expected_denominator
):
    assert expression_from_sympy(sympy_power) == FunctionCall(
        "div", (1, expected_denominator)
    )


@pytest.mark.parametrize(
    "sympy_function_call, expected_function_call",
    [
        (sympy.cos(2), FunctionCall("cos", (2,))),
        (sympy.sin(sympy.Symbol("theta")), FunctionCall("sin", (Symbol("theta"),))),
        (sympy.exp(sympy.Symbol("x")), FunctionCall("exp", (Symbol("x"),))),
    ],
)
def test_sympy_function_calls_are_converted_to_function_call_object_with_appropriate_function_name(
    sympy_function_call, expected_function_call
):
    assert expression_from_sympy(sympy_function_call) == expected_function_call
