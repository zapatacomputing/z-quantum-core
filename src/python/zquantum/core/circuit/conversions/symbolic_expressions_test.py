"""Test cases for symbolic_expressions module."""
import sympy
import pytest
from .symbolic_expressions import expression_tree_from_sympy, Symbol


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
