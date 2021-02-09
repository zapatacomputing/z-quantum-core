import pytest
import qiskit

from .expressions import FunctionCall, Symbol
from .qiskit_expressions import expression_from_qiskit, integer_pow, QISKIT_DIALECT
from .translations import translate_expression


THETA = qiskit.circuit.Parameter("theta")
PHI = qiskit.circuit.Parameter("phi")


EQUIVALENT_EXPRESSIONS = [
    (
        FunctionCall(
            name="add",
            args=(1, FunctionCall(name="mul", args=(2, Symbol(name="theta")))),
        ),
        THETA * 2 + 1,
    ),
    (
        FunctionCall(
            name="add",
            args=(1, FunctionCall(name="pow", args=(Symbol(name="theta"), 2))),
        ),
        THETA * THETA + 1,
    ),
    (
        FunctionCall(
            name="sub",
            args=(
                2,
                FunctionCall(
                    name="mul", args=(Symbol(name="phi"), Symbol(name="theta"))
                ),
            ),
        ),
        2 - THETA * PHI,
    ),
]

INTERMEDIATE_EXPRESSIONS = [expr for expr, _ in EQUIVALENT_EXPRESSIONS]


class TestParsingQiskitExpressions:
    @pytest.mark.parametrize("intermediate_expr, qiskit_expr", EQUIVALENT_EXPRESSIONS)
    def test_parsed_intermediate_expression_matches_equivalent_expression(
        self, intermediate_expr, qiskit_expr
    ):
        parsed_expr = expression_from_qiskit(qiskit_expr)
        assert parsed_expr == intermediate_expr

    @pytest.mark.parametrize("expr", INTERMEDIATE_EXPRESSIONS)
    def test_translate_parse_identity(self, expr):
        # NOTE: the other way round (Qiskit -> intermediate -> Qiskit) can't be done
        # directly, because Qiskit expressions don't implement equality checks.
        qiskit_expr = translate_expression(expr, QISKIT_DIALECT)
        parsed_expr = expression_from_qiskit(qiskit_expr)
        assert parsed_expr == expr


class TestIntegerPower:
    def test_only_integer_exponents_are_valid_for_integer_power(self):
        with pytest.raises(ValueError):
            integer_pow(2, 2.5)

    @pytest.mark.parametrize("base", [10, THETA])
    def test_integer_power_with_exponent_0_is_equal_to_one(self, base):
        assert integer_pow(base, 0) == 1

    @pytest.mark.parametrize(
        "base, exponent, expected_result",
        [(2.5, 3, 2.5 ** 3), (THETA, 2, THETA * THETA)],
    )
    def test_integer_power_with_positive_exponent_is_converted_to_repeated_multiplication(
        self, base, exponent, expected_result
    ):
        assert integer_pow(base, exponent) == expected_result

    def test_negative_exponent_cannot_be_used_if_base_is_zero(self):
        with pytest.raises(ValueError):
            integer_pow(0, -10)

    @pytest.mark.parametrize(
        "base, exponent, expected_result",
        [(2.0, -4, 0.5 ** 4), (THETA, -3, (1 / THETA) * (1 / THETA) * (1 / THETA))],
    )
    def test_integer_power_with_negative_exponent_is_converted_to_repeated_multiplication_of_reciprocals(
        self, base, exponent, expected_result
    ):
        assert integer_pow(base, exponent) == expected_result
