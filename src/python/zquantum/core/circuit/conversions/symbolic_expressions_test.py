"""Test cases for symbolic_expressions module."""
import sympy
import pytest
from .symbolic_expressions import expression_tree_from_sympy, Symbol, FunctionCall


class TestBuildingTreeFromSympyExpression:
    @pytest.mark.parametrize(
        "sympy_symbol, expected_symbol",
        [
            (sympy.Symbol("theta"), Symbol("theta")),
            (sympy.Symbol("x"), Symbol("x")),
            (sympy.Symbol("c_i"), Symbol("c_i")),
        ],
    )
    def test_symbols_are_converted_to_instance_of_symbol_class(
        self, sympy_symbol, expected_symbol
    ):
        assert expression_tree_from_sympy(sympy_symbol) == expected_symbol

    @pytest.mark.parametrize(
        "sympy_number, expected_number",
        [
            (sympy.sympify(2), 2),
            (sympy.sympify(-2.5), -2.5),
            (sympy.Rational(3, 8), 0.375),
        ],
    )
    def test_sympy_numbers_are_converted_to_corresponding_native_number(
        self, sympy_number, expected_number
    ):
        assert expression_tree_from_sympy(sympy_number) == expected_number

    def test_imaginary_unit_is_converted_to_1j(self):
        assert expression_tree_from_sympy(sympy.I) == 1j

    # In below methods we explicitly construct Add and Mul objects
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
        self, sympy_add, expected_args
    ):
        assert expression_tree_from_sympy(sympy_add) == FunctionCall(
            "add", expected_args
        )

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
        self, sympy_mul, expected_args
    ):
        assert expression_tree_from_sympy(sympy_mul) == FunctionCall(
            "mul", expected_args
        )
