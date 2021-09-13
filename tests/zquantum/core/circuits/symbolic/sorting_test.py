import pytest
import sympy
from zquantum.core.circuits import natural_key, natural_key_revlex


@pytest.mark.parametrize(
    "unordered_symbols, expected_ordered_symbols",
    [
        (
            sympy.symbols("theta_10, theta_2, theta_1"),
            sympy.symbols("theta_1, theta_2, theta_10"),
        ),
        (
            sympy.symbols("beta_10, theta_1, beta_2, theta_2"),
            sympy.symbols("beta_2, beta_10, theta_1, theta_2"),
        ),
        (sympy.symbols("gamma, beta, alpha"), sympy.symbols("alpha, beta, gamma")),
    ],
)
def test_natural_key_orders_symbols_as_expected(
    unordered_symbols, expected_ordered_symbols
):
    assert sorted(unordered_symbols, key=natural_key) == list(expected_ordered_symbols)


@pytest.mark.parametrize(
    "unordered_symbols, expected_ordered_symbols",
    [
        (
            sympy.symbols("theta_10, theta_2, theta_1"),
            sympy.symbols("theta_1, theta_2, theta_10"),
        ),
        (
            sympy.symbols("beta_10, theta_1, beta_2, theta_2"),
            sympy.symbols("theta_1, beta_2, theta_2, beta_10"),
        ),
        (sympy.symbols("gamma, beta, alpha"), sympy.symbols("alpha, beta, gamma")),
    ],
)
def test_natural_key_revlex_orders_symbols_as_expected(
    unordered_symbols, expected_ordered_symbols
):
    assert sorted(unordered_symbols, key=natural_key_revlex) == list(
        expected_ordered_symbols
    )
