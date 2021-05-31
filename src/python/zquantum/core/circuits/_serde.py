import json
from functools import singledispatch
from typing import Iterable, List, Mapping

import sympy
from zquantum.core.typing import DumpTarget, LoadSource

from ..serialization import ensure_open
from ..utils import SCHEMA_VERSION
from . import _builtin_gates, _circuit, _gates

CIRCUIT_SCHEMA = SCHEMA_VERSION + "-circuit-v2"
CIRCUITSET_SCHEMA = SCHEMA_VERSION + "-circuitset-v2"


def serialize_expr(expr: sympy.Expr):
    return str(expr)


def _make_symbols_map(symbol_names):
    return {name: sympy.Symbol(name) for name in symbol_names}


def deserialize_expr(expr_str, symbol_names):
    symbols_map = _make_symbols_map(symbol_names)
    return sympy.sympify(expr_str, locals=symbols_map)


def builtin_gate_by_name(name):
    return _builtin_gates.builtin_gate_by_name(name)


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


def _map_eager(fn, iterable: Iterable):
    return list(map(fn, iterable))


# ---------- serialization ----------


@singledispatch
def to_dict(obj):
    raise NotImplementedError(f"Serialization isn't implemented for {type(obj)}")


@to_dict.register
def _circuit_to_dict(circuit: _circuit.Circuit):
    """
    Returns:
        A mapping with keys:
            - "schema"
            - "n_qubits"
            - "symbolic_params"
            - "gates"
    """
    custom_gate_definitions = circuit.collect_custom_gate_definitions()
    return {
        "schema": CIRCUIT_SCHEMA,
        "n_qubits": circuit.n_qubits,
        **(
            {
                "operations": _map_eager(to_dict, circuit.operations),
            }
            if circuit.operations
            else {}
        ),
        **(
            {
                "custom_gate_definitions": _map_eager(to_dict, custom_gate_definitions),
            }
            if custom_gate_definitions
            else {}
        ),
    }


@to_dict.register(list)
def _circuitset_to_dict(circuitset: List[_circuit.Circuit]) -> Mapping:
    """
    Returns:
        A mapping with keys:
            - "schema"
            - "circuits" - list of circuits in this circuitset
    """
    return {
        "schema": CIRCUITSET_SCHEMA,
        "circuits": _map_eager(_circuit_to_dict, circuitset),
    }


@to_dict.register
def _gate_operation_to_dict(gate_operation: _gates.GateOperation):
    return {
        "type": "gate_operation",
        "gate": to_dict(gate_operation.gate),
        "qubit_indices": list(gate_operation.qubit_indices),
    }


@to_dict.register
def _basic_gate_to_dict(gate: _gates.MatrixFactoryGate):
    return {
        "name": gate.name,
        **({"params": _map_eager(serialize_expr, gate.params)} if gate.params else {}),
        **(
            {"free_symbols": sorted(map(str, gate.free_symbols))}
            if gate.free_symbols
            else {}
        ),
    }


@to_dict.register
def _custom_gate_def_to_dict(gate_def: _gates.CustomGateDefinition):
    return {
        "gate_name": gate_def.gate_name,
        "matrix": _matrix_to_json(gate_def.matrix),
        "params_ordering": _map_eager(serialize_expr, gate_def.params_ordering),
    }


@to_dict.register
def _controlled_gate_to_dict(gate: _gates.ControlledGate):
    return {
        "name": gate.name,
        "wrapped_gate": to_dict(gate.wrapped_gate),
        "num_control_qubits": gate.num_control_qubits,
    }


@to_dict.register
def _dagger_gate_to_dict(gate: _gates.Dagger):
    return {
        "name": gate.name,
        "wrapped_gate": to_dict(gate.wrapped_gate),
    }


# ---------- deserialization ----------


def circuit_from_dict(dict_):
    schema = dict_.get("schema")
    if schema != CIRCUIT_SCHEMA:
        raise ValueError(f"Invalid circuit schema: {schema}")

    defs = [
        custom_gate_def_from_dict(def_dict)
        for def_dict in dict_.get("custom_gate_definitions", [])
    ]
    return _circuit.Circuit(
        operations=[
            _gate_operation_from_dict(op_dict, defs)
            for op_dict in dict_.get("operations", [])
        ],
        n_qubits=dict_["n_qubits"],
    )


def _gate_operation_from_dict(dict_, custom_gate_defs):
    return _gates.GateOperation(
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


def _builtin_gate_from_dict(dict_) -> _builtin_gates.GateRef:
    gate_ref = builtin_gate_by_name(dict_["name"])
    if gate_ref is None:
        raise KeyError()

    if _gates.gate_is_parametric(gate_ref, dict_.get("params")):
        return gate_ref(
            *[
                deserialize_expr(param, dict_.get("free_symbols", []))
                for param in dict_["params"]
            ]
        )
    else:
        return gate_ref


def _special_gate_from_dict(dict_, custom_gate_defs) -> _gates.Gate:
    if dict_["name"] == _gates.CONTROLLED_GATE_NAME:
        wrapped_gate = _gate_from_dict(dict_["wrapped_gate"], custom_gate_defs)
        return _gates.ControlledGate(wrapped_gate, dict_["num_control_qubits"])

    elif dict_["name"] == _gates.DAGGER_GATE_NAME:
        wrapped_gate = _gate_from_dict(dict_["wrapped_gate"], custom_gate_defs)
        return _gates.Dagger(wrapped_gate)

    else:
        raise KeyError()


def custom_gate_def_from_dict(dict_) -> _gates.CustomGateDefinition:
    symbols = [sympy.Symbol(term) for term in dict_.get("params_ordering", [])]
    return _gates.CustomGateDefinition(
        gate_name=dict_["gate_name"],
        matrix=_matrix_from_json(dict_["matrix"], dict_.get("params_ordering", [])),
        params_ordering=tuple(symbols),
    )


def _custom_gate_instance_from_dict(dict_, custom_gate_defs) -> _gates.Gate:
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


def circuitset_from_dict(dict_) -> List[_circuit.Circuit]:
    schema = dict_.get("schema")
    if schema != CIRCUITSET_SCHEMA:
        raise ValueError(f"Invalid circuit schema: {schema}")

    return _map_eager(circuit_from_dict, dict_["circuits"])


def load_circuit(load_src: LoadSource):
    with ensure_open(load_src) as f:
        return circuit_from_dict(json.load(f))


def load_circuitset(load_src: LoadSource):
    with ensure_open(load_src) as f:
        return circuitset_from_dict(json.load(f))


def save_circuit(circuit: _circuit.Circuit, dump_target: DumpTarget):
    with ensure_open(dump_target, "w") as f:
        json.dump(to_dict(circuit), f)


def save_circuitset(circuitset: List[_circuit.Circuit], dump_target: DumpTarget):
    with ensure_open(dump_target, "w") as f:
        json.dump(to_dict(circuitset), f)
