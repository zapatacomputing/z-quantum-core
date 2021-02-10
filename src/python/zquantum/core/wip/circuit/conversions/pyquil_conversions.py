"""Utilities for converting gates and circuits to and from Pyquil objects."""
from copy import copy
from functools import singledispatch
import math
from typing import Union, Optional, overload, Iterable
import numpy as np
import pyquil
import pyquil.gates
import sympy

from .. import Circuit, Gate, ControlledGate
from ..gates import (
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
    XY
)
from .symbolic.sympy_expressions import expression_from_sympy
from .symbolic.translations import translate_expression
from .symbolic.pyquil_expressions import QUIL_DIALECT, expression_from_pyquil
from .symbolic.sympy_expressions import SYMPY_DIALECT


ORQUESTRA_CLS_TO_PYQUIL_FUNCTION = {
    X: pyquil.gates.X,
    Y: pyquil.gates.Y,
    Z: pyquil.gates.Z,
    I: pyquil.gates.I,
    T: pyquil.gates.T,
    H: pyquil.gates.H,
    RX: pyquil.gates.RX,
    RY: pyquil.gates.RY,
    RZ: pyquil.gates.RZ,
    PHASE: pyquil.gates.PHASE,
    CZ: pyquil.gates.CZ,
    CNOT: pyquil.gates.CNOT,
    SWAP: pyquil.gates.SWAP,
    CPHASE: pyquil.gates.CPHASE,
    XY: pyquil.gates.XY
}


# A Mapping from PyQuil gate names to the Orquestra classes.
PYQUIL_NAME_TO_ORQUESTRA_CLS = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "T": T,
    "I": I,
    "H": H,
    "RX": RX,
    "RY": RY,
    "RZ": RZ,
    "PHASE": PHASE,
    "CZ": CZ,
    "CNOT": CNOT,
    "CPHASE": CPHASE,
    "SWAP": SWAP,
    "XY": XY
}


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
    required_declarations = (
        pyquil.quil.Declare(str(param), "REAL") for param in gate.symbolic_params
    )
    for declaration in required_declarations:
        if declaration not in program.instructions:
            program += declaration
    return _convert_gate_to_pyquil(gate, program)


def _convert_ordinary_gate_to_pyquil(
    gate: Gate, _program: Optional[pyquil.Program] = None
) -> pyquil.gates.Gate:
    pyquil_function = ORQUESTRA_CLS_TO_PYQUIL_FUNCTION[type(gate)]
    translated_params = [
        translate_expression(expression_from_sympy(param), QUIL_DIALECT)
        for param in gate.params
    ]
    return pyquil_function(*translated_params, *gate.qubits)


@singledispatch
def _convert_gate_to_pyquil(
    gate: Gate, _program: Optional[pyquil.Program] = None
) -> pyquil.gates.Gate:
    try:
        return _convert_ordinary_gate_to_pyquil(gate, _program)
    except KeyError:
        raise NotImplementedError(f"Cannot convert gate {gate} to PyQUil.")


@_convert_gate_to_pyquil.register
def convert_controlled_gate_to_pyquil(
    gate: ControlledGate, _program: Optional[pyquil.Program]
) -> pyquil.gates.Gate:
    if type(gate) in (CZ, CNOT, CPHASE):
        return _convert_ordinary_gate_to_pyquil(gate, _program)
    else:
        return convert_to_pyquil(gate.target_gate, _program).controlled(gate.control)


@_convert_gate_to_pyquil.register
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
def convert_from_pyquil(
    obj: Union[pyquil.Program, pyquil.quil.Gate], custom_gates=None
):
    raise NotImplementedError(
        f"Conversion from pyquil to orquestra not implemented for {obj}"
    )


def custom_gate_factory_from_pyquil_defgate(gate: pyquil.quil.DefGate):
    num_qubits = int(math.log(gate.matrix.shape[0], 2))
    assert 2 ** num_qubits == gate.matrix.shape[0]

    sympy_matrix = sympy.Matrix(
        [
            [
                translate_expression(expression_from_pyquil(element), SYMPY_DIALECT)
                for element in row
            ]
            for row in gate.matrix.tolist()
        ]
    )

    # Order of parameters in our CustomGates may vary. On the contrary,
    # order of parameters in pyquil is fixed.
    # Therefore we remember this order so we can later correctly evaluate
    # our custom gate.

    symbols = [param.name for param in gate.parameters] if gate.parameters else None

    def _factory(*args):
        qubits = args[:num_qubits]
        orquestra_gate = CustomGate(sympy_matrix, qubits=tuple(qubits), name=gate.name)
        if len(args) != num_qubits:
            parameters = args[num_qubits:]

            orquestra_gate = orquestra_gate.evaluate(
                {symbol: value for symbol, value in zip(symbols, parameters)}
            )
        return orquestra_gate

    return _factory


@convert_from_pyquil.register
def convert_gate_from_pyquil(gate: pyquil.quil.Gate, custom_gates=None) -> Gate:
    number_of_control_modifiers = gate.modifiers.count("CONTROLLED")

    all_qubits = pyquil_qubits_to_numbers(gate.qubits)
    control_qubits = all_qubits[:number_of_control_modifiers]
    target_qubits = all_qubits[number_of_control_modifiers:]

    orquestra_params = tuple(
        translate_expression(expression_from_pyquil(param), SYMPY_DIALECT)
        for param in gate.params
    )

    pyquil_name_to_orquestra_cls = copy(PYQUIL_NAME_TO_ORQUESTRA_CLS)

    if custom_gates is not None:
        pyquil_name_to_orquestra_cls.update(custom_gates)

    try:
        gate_cls = pyquil_name_to_orquestra_cls[gate.name]
        result = gate_cls(*target_qubits, *orquestra_params)

        # Control qubits need to be applied in reverse because in PyQuil they
        # are prepended to the list when applying control modifier.
        for qubit in reversed(control_qubits):
            result = ControlledGate(result, qubit)

        # PyQuil allows multiple DAGGER modifiers in gate's definition.
        # Since two daggers cancel out, we only need to apply it if
        # the total number of DAGGER modifiers is odd.
        if gate.modifiers.count("DAGGER") % 2 == 1:
            result = result.dagger

        return result
    except TypeError:
        raise ValueError(
            f"Cannot convert {gate}. Please check that you haven't reimplemented "
            "predefined gate. If this is not the case, contact Orquestra support."
        )
    except KeyError:
        raise ValueError(
            f"Conversion to Orquestra is not supported for {gate.name} gate. "
            "If this is a custom gate, make sure to convert it together with "
            "a corresponding PyQuil program."
        )


@convert_from_pyquil.register
def convert_pyquil_program_to_orquestra(program: pyquil.Program, custom_gates=None):
    custom_gates = {
        definition.name: custom_gate_factory_from_pyquil_defgate(definition)
        for definition in program.defined_gates
    }

    gates_in_program = [
        instruction
        for instruction in program.instructions
        if isinstance(instruction, pyquil.quil.Gate)
    ]

    return Circuit(
        [convert_from_pyquil(gate, custom_gates) for gate in gates_in_program]
    )
