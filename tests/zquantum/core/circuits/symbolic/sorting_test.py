################################################################################
# © Copyright 2021 Zapata Computing Inc.
################################################################################
import pytest
import sympy
from zquantum.core.circuits import natural_key, natural_key_revlex
from zquantum.core.circuits.symbolic import natural_key_fixed_names_order


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


@pytest.mark.parametrize(
    "names_order, unordered_symbols, expected_ordered_symbols",
    [
        (
            ("gamma", "beta"),
            sympy.symbols("beta_0, gamma_0, beta_3, gamma_3, beta_4"),
            sympy.symbols("gamma_0, beta_0, gamma_3, beta_3, beta_4"),
        ),
        (
            ("theta", "beta", "gamma"),
            sympy.symbols("gamma_0, beta_0, theta_0, beta_1, theta_1, gamma_1"),
            sympy.symbols("theta_0, beta_0, gamma_0, theta_1, beta_1, gamma_1"),
        ),
    ],
)
def test_natural_key_fixed_names_order_orders_symbols_ax_expected(
    names_order, unordered_symbols, expected_ordered_symbols
):
    key = natural_key_fixed_names_order(names_order)
    assert sorted(unordered_symbols, key=key) == list(expected_ordered_symbols)
