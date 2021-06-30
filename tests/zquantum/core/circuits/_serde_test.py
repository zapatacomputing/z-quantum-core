import io

import numpy as np
import pytest
import sympy
from zquantum.core.circuits import _builtin_gates, _circuit, _gates
from zquantum.core.circuits._serde import (
    circuit_from_dict,
    circuitset_from_dict,
    custom_gate_def_from_dict,
    deserialize_expr,
    load_circuit,
    load_circuitset,
    save_circuit,
    save_circuitset,
    serialize_expr,
    to_dict,
)

ALPHA = sympy.Symbol("alpha")
GAMMA = sympy.Symbol("gamma")
THETA = sympy.Symbol("theta")


CUSTOM_U_GATE = _gates.CustomGateDefinition(
    "U",
    sympy.Matrix(
        [
            [THETA, GAMMA],
            [-GAMMA, THETA],
        ]
    ),
    (THETA, GAMMA),
)


EXAMPLE_CIRCUITS = [
    _circuit.Circuit(),
    _circuit.Circuit([_builtin_gates.X(0)]),
    _circuit.Circuit([_builtin_gates.X(2), _builtin_gates.Y(1)]),
    _circuit.Circuit(
        [
            _builtin_gates.H(0),
            _builtin_gates.CNOT(0, 1),
            _builtin_gates.RX(0)(5),
            _builtin_gates.RX(np.pi)(2),
        ]
    ),
    _circuit.Circuit(
        [
            _builtin_gates.RX(GAMMA * 2)(3),
        ]
    ),
    _circuit.Circuit(
        operations=[
            _builtin_gates.T(0),
            CUSTOM_U_GATE(1, -1)(3),
            CUSTOM_U_GATE(ALPHA, -1)(2),
        ],
    ),
    _circuit.Circuit(
        operations=[
            CUSTOM_U_GATE(2 + 3j, -1)(2),
        ],
    ),
    _circuit.Circuit(
        [
            _builtin_gates.H.controlled(1)(0, 1),
        ]
    ),
    _circuit.Circuit(
        [
            _builtin_gates.Z.controlled(2)(4, 3, 0),
        ]
    ),
    _circuit.Circuit(
        [
            _builtin_gates.RY(ALPHA * GAMMA).controlled(1)(3, 2),
        ]
    ),
    _circuit.Circuit(
        [
            _builtin_gates.X.dagger(2),
            _builtin_gates.I.dagger(4),
            _builtin_gates.Y.dagger(1),
            _builtin_gates.Z.dagger(2),
            _builtin_gates.T.dagger(7),
        ]
    ),
    _circuit.Circuit(
        [
            _builtin_gates.RX(-np.pi).dagger(2),
            _builtin_gates.RY(-np.pi / 2).dagger(1),
            _builtin_gates.RZ(0).dagger(0),
            _builtin_gates.PHASE(np.pi / 5).dagger(2),
        ]
    ),
    _circuit.Circuit(
        [
            _builtin_gates.RX(GAMMA * ALPHA).dagger(1),
        ]
    ),
]


@pytest.mark.parametrize(
    "circuit",
    EXAMPLE_CIRCUITS,
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


class TestCircuitsetSerialization:
    @pytest.mark.parametrize(
        "circuitset",
        [
            [],
            list(EXAMPLE_CIRCUITS),
        ],
    )
    def test_roundrip_results_in_same_circuitset(self, circuitset):
        serialized = to_dict(circuitset)
        assert circuitset_from_dict(serialized) == circuitset

    @pytest.mark.parametrize(
        "dict_",
        [
            {},
            to_dict(_circuit.Circuit()),
        ],
    )
    def test_raises_error_with_invalid_dict(self, dict_):
        with pytest.raises(ValueError):
            circuitset_from_dict(dict_)


class TestCustomGateDefinitionSerialization:
    @pytest.mark.parametrize(
        "gate_def",
        [
            _gates.CustomGateDefinition(
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


class TestIOHelpers:
    @pytest.mark.parametrize("circuit", EXAMPLE_CIRCUITS)
    def test_load_save_circuit_roundtrip(self, circuit):
        buf = io.StringIO()
        save_circuit(circuit, buf)
        buf.seek(0)
        assert load_circuit(buf) == circuit

    def test_load_save_circuitset_roundtrip(self):
        circuitset = EXAMPLE_CIRCUITS
        buf = io.StringIO()
        save_circuitset(circuitset, buf)
        buf.seek(0)
        assert load_circuitset(buf) == circuitset
