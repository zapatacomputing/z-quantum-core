"""Test cases for symbolic_expressions module."""
import sympy
import pytest
from .symbolic_expressions import expression_tree_from_sympy, Symbol


class TestBuildingTreeFromSympyExpression:
    @pytest.mark.parametrize(
        "sympy_symbol,expected_symbol",
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
