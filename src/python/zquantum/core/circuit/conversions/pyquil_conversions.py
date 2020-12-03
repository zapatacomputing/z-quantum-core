"""Utilities for converting gates and circuits to and from Pyquil objects."""
from functools import singledispatch
from itertools import chain
from typing import Union, Optional, overload, Iterable
import numpy as np
import pyquil
import pyquil.gates
from ...circuit import Gate, ControlledGate
from ..circuit import Circuit
from ...circuit.gates import (
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    PHASE,
    T,
    I,
    H,
    Dagger,
    CZ,
    CNOT,
    CPHASE,
    SWAP,
    CustomGate,
)
from .symbolic.sympy_expressions import expression_from_sympy
from .symbolic.translations import translate_expression
from .symbolic.pyquil_expressions import QUIL_DIALECT, expression_from_pyquil
from .symbolic.sympy_expressions import SYMPY_DIALECT


SINGLE_QUBIT_NONPARAMETRIC_GATES = {
    X: pyquil.gates.X,
    Y: pyquil.gates.Y,
    Z: pyquil.gates.Z,
    I: pyquil.gates.I,
    T: pyquil.gates.T,
    H: pyquil.gates.H,
}


ROTATION_GATES = {
    RX: pyquil.gates.RX,
    RY: pyquil.gates.RY,
    RZ: pyquil.gates.RZ,
    PHASE: pyquil.gates.PHASE,
}


TWO_QUBIT_CONTROLLED_GATES = {
    CZ: pyquil.gates.CZ,
    CNOT: pyquil.gates.CNOT,
}


PYQUIL_NAME_TO_ORQUESTRA_CLS = {
    cls.__name__: cls
    for cls in chain(
        SINGLE_QUBIT_NONPARAMETRIC_GATES,
        TWO_QUBIT_CONTROLLED_GATES,
        ROTATION_GATES
    )
}

PYQUIL_NAME_TO_ORQUESTRA_CLS["CPHASE"] = CPHASE
PYQUIL_NAME_TO_ORQUESTRA_CLS["SWAP"] = SWAP


def pyquil_qubits_to_numbers(qubits: Iterable[pyquil.quil.Qubit]):
    return tuple(qubit.index for qubit in qubits)


@overload
def convert_to_pyquil(obj: Circuit) -> pyquil.Program:
    pass


@overload
def convert_to_pyquil(
    obj: Gate, program: Optional[pyquil.Program] = None
) -> pyquil.quil.Gate:
    pass


@singledispatch
def convert_to_pyquil(obj, program: Optional[pyquil.Program] = None):
    raise NotImplementedError(f"Cannot convert {obj} to PyQuil object.")


@convert_to_pyquil.register
def convert_gate_to_pyquil(
    gate: Gate, program: Optional[pyquil.Program] = None
) -> pyquil.gates.Gate:
    required_declarations = (pyquil.quil.Declare(str(param), "REAL") for param in gate.symbolic_params)
    for declaration in required_declarations:
        if declaration not in program.instructions:
            program += declaration
    return _convert_gate_to_pyquil(gate, program)


@singledispatch
def _convert_gate_to_pyquil(
    gate: Gate, _program: Optional[pyquil.Program] = None
) -> pyquil.gates.Gate:
    raise NotImplementedError(f"Cannot convert gate {gate} to PyQUil.")


@_convert_gate_to_pyquil.register(X)
@_convert_gate_to_pyquil.register(Y)
@_convert_gate_to_pyquil.register(Z)
@_convert_gate_to_pyquil.register(I)
@_convert_gate_to_pyquil.register(T)
@_convert_gate_to_pyquil.register(H)
def convert_single_qubit_nonparametric_gate_to_pyquil(
    gate: Union[X, Y, Z], _program: Optional[pyquil.Program] = None
) -> pyquil.gates.Gate:
    return SINGLE_QUBIT_NONPARAMETRIC_GATES[type(gate)](gate.qubit)


@_convert_gate_to_pyquil.register(RX)
@_convert_gate_to_pyquil.register(RY)
@_convert_gate_to_pyquil.register(RZ)
@_convert_gate_to_pyquil.register(PHASE)
def convert_single_qubit_rotation_gate_to_pyquil(
    gate: Union[RX, RY, RZ, PHASE], _program: Optional[pyquil.Program] = None
) -> pyquil.gates.Gate:
    return ROTATION_GATES[type(gate)](
        translate_expression(expression_from_sympy(gate.angle), QUIL_DIALECT),
        gate.qubit,
    )


@_convert_gate_to_pyquil.register(CNOT)
@_convert_gate_to_pyquil.register(CZ)
@_convert_gate_to_pyquil.register(SWAP)
def convert_two_qubit_nonparametric_gate_to_pyquil(
    gate: Union[CZ], _program: Optional[pyquil.Program] = None
) -> pyquil.gates.Gate:
    return TWO_QUBIT_CONTROLLED_GATES[type(gate)](*gate.qubits)


@_convert_gate_to_pyquil.register(CPHASE)
def convert_CPHASE_to_pyquil(
    gate: CPHASE, _program: Optional[pyquil.Program]
) -> pyquil.gates.Gate:
    return pyquil.gates.CPHASE(
        translate_expression(expression_from_sympy(gate.angle), QUIL_DIALECT),
        *gate.qubits,
    )


@_convert_gate_to_pyquil.register(SWAP)
def convert_SWAP_gate_to_pyquil(
    gate: SWAP, _program: Optional[pyquil.Program]
) -> pyquil.gates.Gate:
    return pyquil.gates.SWAP(*gate.qubits)


@_convert_gate_to_pyquil.register(ControlledGate)
def convert_controlled_gate_to_pyquil(
    gate: ControlledGate, _program: Optional[pyquil.Program]
) -> pyquil.gates.Gate:
    return convert_to_pyquil(gate.target_gate, _program).controlled(gate.control)


@_convert_gate_to_pyquil.register(Dagger)
def convert_dagger_to_pyquil(
    gate: Dagger, _program: Optional[pyquil.Program]
) -> pyquil.gates.Gate:
    return convert_to_pyquil(gate.gate, _program).dagger()


@_convert_gate_to_pyquil.register(CustomGate)
def convert_custom_gate_to_pyquil(
    gate: CustomGate, program: Optional[pyquil.Program]
) -> pyquil.gates.Gate:
    gate_definition = None

    for definition in program.defined_gates:
        if definition.name == gate.name:
            gate_definition = definition
            break

    if gate_definition is None:
        converted_matrix = [
            [
                translate_expression(expression_from_sympy(element), QUIL_DIALECT)
                for element in row
            ]
            for row in gate.matrix.tolist()
        ]
        params = [
            translate_expression(expression_from_sympy(param), QUIL_DIALECT)
            for param in gate.symbolic_params
        ]
        gate_definition = pyquil.quil.DefGate(
            gate.name, np.array(converted_matrix), params
        )
        program += gate_definition

    # Custom defined gates' constructors behave differently depending on
    # whether the gate has parameters or not, hence the below condition.
    if gate_definition.parameters:
        new_gate = gate_definition.get_constructor()(*gate_definition.parameters)(
            *gate.qubits
        )
    else:
        new_gate = gate_definition.get_constructor()(*gate.qubits)

    return new_gate


@convert_to_pyquil.register(Circuit)
def convert_circuit_to_pyquil(
    circuit: Circuit, _program: Optional[pyquil.Program] = None
) -> pyquil.Program:
    program = pyquil.Program()
    for gate in circuit.gates:
        program += convert_to_pyquil(gate, program)

    return program


@singledispatch
def convert_from_pyquil(obj: Union[pyquil.Program, pyquil.quil.Gate]):
    pass


@convert_from_pyquil.register
def convert_gate_from_pyquil(gate: pyquil.quil.Gate) -> Gate:
    try:
        gate_cls = PYQUIL_NAME_TO_ORQUESTRA_CLS[gate.name]
        if (
            gate_cls in SINGLE_QUBIT_NONPARAMETRIC_GATES
            or gate_cls in TWO_QUBIT_CONTROLLED_GATES
        ):
            return gate_cls(*pyquil_qubits_to_numbers(gate.qubits))
        elif gate_cls in ROTATION_GATES or gate_cls == CPHASE:
            return gate_cls(
                *pyquil_qubits_to_numbers(gate.qubits),
                translate_expression(
                    expression_from_pyquil(gate.params[0]),
                    SYMPY_DIALECT
                )
            )
        elif gate_cls == SWAP:
            return gate_cls(pyquil_qubits_to_numbers(gate.qubits))

    except KeyError:
        raise ValueError(
            f"Conversion to Orquestra is not supported for {gate.name} gate"
        )