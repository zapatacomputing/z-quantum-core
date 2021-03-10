import pytest
import sympy
import numpy as np

from . import _gates as g
from . import _builtin_gates as bg
from ._serde import serialize_expr, deserialize_expr, circuit_from_dict, custom_gate_def_from_dict, to_dict


ALPHA = sympy.Symbol("alpha")
GAMMA = sympy.Symbol("gamma")
THETA = sympy.Symbol("theta")


CUSTOM_U_GATE = g.CustomGateDefinition(
    "U",
    sympy.Matrix(
        [
            [THETA, GAMMA],
            [-GAMMA, THETA],
        ]
    ),
    (THETA, GAMMA),
)


@pytest.mark.parametrize(
    "circuit",
    [
        g.Circuit(),
        g.Circuit([bg.X(0)]),
        g.Circuit([bg.X(2), bg.Y(1)]),
        g.Circuit(
            [
                bg.H(0),
                bg.CNOT(0, 1),
                bg.RX(0)(5),
                bg.RX(np.pi)(2),
            ]
        ),
        g.Circuit(
            [
                bg.RX(GAMMA * 2)(3),
            ]
        ),
        g.Circuit(
            operations=[
                bg.T(0),
                CUSTOM_U_GATE(1, -1)(3),
                CUSTOM_U_GATE(ALPHA, -1)(2),
            ],
            custom_gate_definitions=[CUSTOM_U_GATE],
        ),
        g.Circuit(
            operations=[
                CUSTOM_U_GATE(2 + 3j, -1)(2),
            ],
            custom_gate_definitions=[CUSTOM_U_GATE],
        ),
        g.Circuit(
            [
                bg.H.controlled(1)(0, 1),
            ]
        ),
        g.Circuit(
            [
                bg.Z.controlled(2)(4, 3, 0),
            ]
        ),
        g.Circuit(
            [
                bg.RY(ALPHA * GAMMA).controlled(1)(3, 2),
            ]
        ),
        g.Circuit(
            [
                bg.X.dagger(2),
                bg.I.dagger(4),
                bg.Y.dagger(1),
                bg.Z.dagger(2),
                bg.T.dagger(7),
            ]
        ),
        g.Circuit(
            [
                bg.RX(-np.pi).dagger(2),
                bg.RY(-np.pi / 2).dagger(1),
                bg.RZ(0).dagger(0),
                bg.PHASE(np.pi / 5).dagger(2),
            ]
        ),
        g.Circuit(
            [
                bg.RX(GAMMA * ALPHA).dagger(1),
            ]
        ),
    ],
)
class TestCircuitSerialization:
    def test_roundrip_results_in_same_circuit(self, circuit):
        serialized = to_dict(circuit)
        assert circuit_from_dict(serialized) == circuit

    def test_deserialized_gates_produce_matrices(self, circuit):
        deserialized_circuit = circuit_from_dict(to_dict(circuit))
        for operation in deserialized_circuit.operations:
            # matrices are computed lazily, so we have to call the getter to know if
            # we deserialized parameters properly
            operation.gate.matrix


class TestCustomGateDefinitionSerialization:
    @pytest.mark.parametrize(
        "gate_def",
        [
            g.CustomGateDefinition(
                "V", sympy.Matrix([[THETA, GAMMA], [-GAMMA, THETA]]), (THETA, GAMMA)
            )
        ],
    )
    def test_roundtrip_gives_back_same_def(self, gate_def):
        dict_ = to_dict(gate_def)
        assert custom_gate_def_from_dict(dict_) == gate_def


class TestExpressionSerialization:
    @pytest.mark.parametrize(
        "expr,symbol_names",
        [
            (0, []),
            (1, []),
            (-1, []),
            (THETA, ["theta"]),
            (GAMMA, ["gamma"]),
            (THETA * GAMMA + 1, ["gamma", "theta"]),
            (2 + 3j, []),
            ((-1 + 2j) * THETA * GAMMA, ["gamma", "theta"]),
        ],
    )
    def test_roundtrip_results_in_equivalent_expression(self, expr, symbol_names):
        serialized = serialize_expr(expr)
        deserialized = deserialize_expr(serialized, symbol_names)
        # `deserialized == expr` wouldn't work here for complex literals because of
        # how Sympy compares expressions
        assert deserialized - expr == 0
