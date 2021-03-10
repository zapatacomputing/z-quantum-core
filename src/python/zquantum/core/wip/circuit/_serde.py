from typing import List, Iterable
from functools import singledispatch

import sympy

from . import _gates as g
from . import _builtin_gates as bg
from ...utils import SCHEMA_VERSION


def serialize_expr(expr):
    return str(expr)


def _make_symbols_map(symbol_names):
    return {name: sympy.Symbol(name) for name in symbol_names}


def deserialize_expr(expr_str, symbol_names):
    symbols_map = _make_symbols_map(symbol_names)
    return sympy.sympify(expr_str, locals=symbols_map)


def builtin_gate_by_name(name):
    return bg.builtin_gate_by_name(name)


def _matrix_to_json(matrix: sympy.Matrix):
    return [
        [serialize_expr(element) for element in matrix.row(row_i)]
        for row_i in range(matrix.shape[0])
    ]


def _matrix_from_json(
    json_rows: List[List[str]], symbols_names: Iterable[str]
) -> sympy.Matrix:
    return sympy.Matrix(
        [
            [deserialize_expr(element, symbols_names) for element in json_row]
            for json_row in json_rows
        ]
    )


CIRCUIT_SCHEMA = SCHEMA_VERSION + "-circuit"


def _mapv(fn, coll):
    return list(map(fn, coll))


# ---------- serialization ----------


@singledispatch
def to_dict(obj):
    raise NotImplementedError(f"Serialization isn't implemented for {type(obj)}")


@to_dict.register
def _circuit_to_dict(circuit: g.Circuit):
    """
    Returns:
        A mapping with keys:
            - "schema"
            - "n_qubits"
            - "symbolic_params"
            - "gates"
    """
    return {
        "schema": CIRCUIT_SCHEMA,
        "n_qubits": circuit.n_qubits,
        **(
            {
                "operations": _mapv(to_dict, circuit.operations),
            }
            if circuit.operations
            else {}
        ),
        **(
            {
                "custom_gate_definitions": _mapv(to_dict, circuit.custom_gate_definitions),
            }
            if circuit.custom_gate_definitions
            else {}
        ),
    }


@to_dict.register
def _gate_operation_to_dict(gate_operation: g.GateOperation):
    return {
        "type": "gate_operation",
        "gate": to_dict(gate_operation.gate),
        "qubit_indices": list(gate_operation.qubit_indices),
    }


@to_dict.register
def _basic_gate_to_dict(gate: g.MatrixFactoryGate):
    return {
        "name": gate.name,
        **(
            {"params": list(map(serialize_expr, gate.params))}
            if gate.params
            else {}
        ),
        **(
            {"free_symbols": sorted(map(str, gate.free_symbols))}
            if gate.free_symbols
            else {}
        ),
    }


@to_dict.register
def _custom_gate_def_to_dict(gate_def: g.CustomGateDefinition):
    return {
        "gate_name": gate_def.gate_name,
        "matrix": _matrix_to_json(gate_def.matrix),
        "params_ordering": _mapv(serialize_expr, gate_def.params_ordering),
    }


@to_dict.register
def _controlled_gate_to_dict(gate: g.ControlledGate):
    return {
        "name": gate.name,
        "wrapped_gate": to_dict(gate.wrapped_gate),
        "num_control_qubits": gate.num_control_qubits,
    }


@to_dict.register
def _dagger_gate_to_dict(gate: g.Dagger):
    return {
        "name": gate.name,
        "wrapped_gate": to_dict(gate.wrapped_gate),
    }


# ---------- deserialization ----------


def circuit_from_dict(dict_):
    defs = [
        custom_gate_def_from_dict(def_dict)
        for def_dict in dict_.get("custom_gate_definitions", [])
    ]
    return g.Circuit(
        operations=[
            _gate_operation_from_dict(op_dict, defs)
            for op_dict in dict_.get("operations", [])
        ],
        n_qubits=dict_["n_qubits"],
        custom_gate_definitions=defs,
    )


def _gate_operation_from_dict(dict_, custom_gate_defs):
    return g.GateOperation(
        gate=_gate_from_dict(dict_["gate"], custom_gate_defs),
        qubit_indices=tuple(dict_["qubit_indices"]),
    )


def _gate_from_dict(dict_, custom_gate_defs):
    """Generic gate deserializer.

    Pass it a JSON dictionary and it'll try its best to return you a proper gate
    instance, regardless of the given gate type."""
    try:
        return _builtin_gate_from_dict(dict_)
    except KeyError:
        pass

    try:
        return _special_gate_from_dict(dict_, custom_gate_defs)
    except KeyError:
        pass

    return _custom_gate_instance_from_dict(dict_, custom_gate_defs)


def _builtin_gate_from_dict(dict_) -> bg.GateRef:
    gate_ref = builtin_gate_by_name(dict_["name"])
    if gate_ref is None:
        raise KeyError()

    if g.gate_is_parametric(gate_ref, dict_.get("params")):
        return gate_ref(
            *[
                deserialize_expr(param, dict_.get("free_symbols", []))
                for param in dict_["params"]
            ]
        )
    else:
        return gate_ref


def _special_gate_from_dict(dict_, custom_gate_defs) -> g.Gate:
    if dict_["name"] == g.CONTROLLED_GATE_NAME:
        wrapped_gate = _gate_from_dict(dict_["wrapped_gate"], custom_gate_defs)
        return g.ControlledGate(wrapped_gate, dict_["num_control_qubits"])

    elif dict_["name"] == g.DAGGER_GATE_NAME:
        wrapped_gate = _gate_from_dict(dict_["wrapped_gate"], custom_gate_defs)
        return g.Dagger(wrapped_gate)

    else:
        raise KeyError()


def custom_gate_def_from_dict(dict_) -> g.CustomGateDefinition:
    symbols = [sympy.Symbol(term) for term in dict_.get("params_ordering", [])]
    return g.CustomGateDefinition(
        gate_name=dict_["gate_name"],
        matrix=_matrix_from_json(dict_["matrix"], dict_.get("params_ordering", [])),
        params_ordering=tuple(symbols),
    )


def _custom_gate_instance_from_dict(dict_, custom_gate_defs) -> g.Gate:
    gate_def = next(
        (
            gate_def
            for gate_def in custom_gate_defs
            if gate_def.gate_name == dict_["name"]
        ),
        None,
    )
    if gate_def is None:
        raise ValueError(
            f"Custom gate definition for {dict_['name']} missing from serialized dict"
        )

    symbol_names = map(serialize_expr, gate_def.params_ordering)
    return gate_def(
        *[deserialize_expr(param, symbol_names) for param in dict_["params"]]
    )
