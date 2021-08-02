"""Test cases for symengine_expressions module."""
import pytest
import symengine
from zquantum.core.circuits.symbolic.expressions import FunctionCall, Symbol
from zquantum.core.circuits.symbolic.symengine_expressions import (
    SYMENGINE_DIALECT,
    expression_from_symengine,
    is_left_addition_of_negation,
    is_multiplication_by_reciprocal,
    is_right_addition_of_negation,
)
from zquantum.core.circuits.symbolic.translations import translate_expression


def _to_symengine(expr):
    return translate_expression(expr, SYMENGINE_DIALECT)


@pytest.mark.parametrize("number", [3, 4.0, 1j, 3.0 - 2j])
def test_native_numbers_are_preserved(number):
    assert expression_from_symengine(number) == number


@pytest.mark.parametrize(
    "symengine_symbol, expected_symbol",
    [
        (symengine.Symbol("theta"), Symbol("theta")),
        (symengine.Symbol("x"), Symbol("x")),
        (symengine.Symbol("c_i"), Symbol("c_i")),
    ],
)
def test_symbols_are_converted_to_instance_of_symbol_class(
    symengine_symbol, expected_symbol
):
    assert expression_from_symengine(symengine_symbol) == expected_symbol


@pytest.mark.parametrize(
    "symengine_number, expected_number, expected_class",
    [
        (symengine.sympify(2), 2, int),
        (symengine.sympify(-2.5), -2.5, float),
        (symengine.Rational(3, 8), 0.375, float),
    ],
)
def test_symengine_numbers_are_converted_to_corresponding_native_number(
    symengine_number, expected_number, expected_class
):
    native_number = expression_from_symengine(symengine_number)
    assert native_number == expected_number
    assert isinstance(native_number, expected_class)


def test_imaginary_unit_is_converted_to_1j():
    assert expression_from_symengine(symengine.I) == 1j


SYMENGINE_EXPRESSIONS = [
    # Add
    1 + symengine.Symbol("x"),
    symengine.Symbol("x") + symengine.Symbol("y") + symengine.Symbol("z"),
    # Mul
    2 * symengine.Symbol("x"),
    symengine.Symbol("x") * symengine.Symbol("y"),
    # Division, also represented as symengine.Mul
    symengine.Symbol("x") / symengine.Symbol("y"),
    symengine.Symbol("x") / (symengine.Symbol("z") + 1),
    # Function calls
    symengine.cos(2),
    symengine.sin(symengine.Symbol("theta")),
    symengine.exp(symengine.Symbol("x")),
]


@pytest.mark.parametrize("symengine_expression", SYMENGINE_EXPRESSIONS)
def test_roundtrip_between_symengine_and_zquantum_expressions(symengine_expression):
    assert (
        _to_symengine(expression_from_symengine(symengine_expression))
        == symengine_expression
    )


@pytest.mark.parametrize(
    "symengine_multiplication",
    [
        symengine.Symbol("x") / symengine.Symbol("y"),
        symengine.Symbol("x") / (symengine.Symbol("z") + 1),
    ],
)
def test_mul_from_division_is_classified_as_multiplication_by_reciprocal(
    symengine_multiplication,
):
    assert is_multiplication_by_reciprocal(symengine_multiplication)


@pytest.mark.parametrize(
    "symengine_multiplication",
    [
        symengine.Symbol("x") * symengine.Symbol("y"),
        2 * symengine.Symbol("theta"),
        symengine.Symbol("x") * symengine.Symbol("y") * symengine.Symbol("z"),
    ],
)
def test_mul_not_from_division_is_not_classified_as_multiplication_by_reciprocal(
    symengine_multiplication,
):
    # Note: obviously you can manually construct multiplication that would
    # be classified as multiplication by reciprocal. The bottom line of this
    # test is: usual, simple multiplications are multiplications, not divisions.
    assert not is_multiplication_by_reciprocal(symengine_multiplication)


@pytest.mark.parametrize(
    "symengine_multiplication",
    [
        symengine.Symbol("x") / symengine.Symbol("y"),
        symengine.Symbol("x") / (symengine.Symbol("z") + 1),
    ],
)
def test_division_is_converted_into_div_fn_call_instead_of_multiplication_by_reciprocal(
    symengine_multiplication,
):
    assert expression_from_symengine(symengine_multiplication).name == "div"


@pytest.mark.parametrize(
    "symengine_addition",
    [
        symengine.Symbol("x") - symengine.Symbol("y"),
        symengine.Symbol("x") - 1 / symengine.Symbol("y"),
    ],
)
def test_add_resulting_from_subtraction_is_classified_as_addition_of_negation(
    symengine_addition,
):
    assert is_left_addition_of_negation(
        symengine_addition
    ) or is_right_addition_of_negation(symengine_addition)


@pytest.mark.parametrize(
    "symengine_addition",
    [symengine.Symbol("x") + symengine.Symbol("y"), symengine.Symbol("x") + 10],
)
def test_add_not_resulting_from_subtraction_is_not_classified_as_addition_of_negation(
    symengine_addition,
):
    assert not (
        is_left_addition_of_negation(symengine_addition)
        or is_right_addition_of_negation(symengine_addition)
    )


@pytest.mark.parametrize(
    "symengine_addition, expected_args",
    [
        (symengine.Symbol("x") - symengine.Symbol("y"), (Symbol("x"), Symbol("y"))),
        (1 - symengine.Symbol("x"), (1, Symbol("x"))),
        (
            1 - symengine.Symbol("x") * symengine.Symbol("y"),
            (1, FunctionCall("mul", (Symbol("x"), Symbol("y")))),
        ),
    ],
)
def test_add_resulting_from_subtraction_is_converted_to_sub_function_call(
    symengine_addition, expected_args
):
    assert expression_from_symengine(symengine_addition) == FunctionCall(
        "sub", expected_args
    )


@pytest.mark.parametrize(
    "symengine_power, expected_args",
    [
        (symengine.Pow(symengine.Symbol("x"), 2), (Symbol("x"), 2)),
        (symengine.Pow(2, symengine.Symbol("x")), (2, Symbol("x"))),
        (
            symengine.Pow(symengine.Symbol("x"), symengine.Symbol("y")),
            (Symbol("x"), Symbol("y")),
        ),
    ],
)
def test_symengine_pow_is_converted_to_pow_function_call(
    symengine_power, expected_args
):
    assert expression_from_symengine(symengine_power) == FunctionCall(
        "pow", expected_args
    )


@pytest.mark.parametrize(
    "symengine_power, expected_denominator",
    [
        (symengine.Pow(symengine.Symbol("x"), -1), Symbol("x")),
        (
            symengine.Pow(
                symengine.Add(
                    symengine.Symbol("x"), symengine.Symbol("y"), evaluate=False
                ),
                -1,
            ),
            FunctionCall("add", (Symbol("x"), Symbol("y"))),
        ),
    ],
)
def test_symengine_power_with_negative_one_exponent_gets_converted_to_division(
    symengine_power, expected_denominator
):
    assert expression_from_symengine(symengine_power).name == "div"


@pytest.mark.parametrize(
    "symengine_function_call, expected_function_call",
    [
        (symengine.cos(2), FunctionCall("cos", (2,))),
        (
            symengine.sin(symengine.Symbol("theta")),
            FunctionCall("sin", (Symbol("theta"),)),
        ),
        (symengine.exp(symengine.Symbol("x")), FunctionCall("exp", (Symbol("x"),))),
    ],
)
def test_symengine_fn_calls_are_converted_to_fn_call_object_with_appropriate_fn_name(
    symengine_function_call, expected_function_call
):
    assert expression_from_symengine(symengine_function_call) == expected_function_call
