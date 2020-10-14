from abc import ABC
from typing import Tuple, Union, Dict, TextIO, Callable
import sympy
from . import Gate, SpecializedGate, X, Z, PHASE


class ControlledGate(SpecializedGate, ABC):
    """Controlled gate."""

    gate_factory: Callable[[int], Gate]

    def __init__(self, control: int, target: int):
        super().__init__((control, target))

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix.diag(
            sympy.eye(2), self.gate_factory(self.qubits[1]).matrix
        )


class CNOT(ControlledGate):
    """Controlled NOT (Controlled X) gate."""

    gate_factory = X


class CZ(ControlledGate):
    """"Controlled Z gate."""

    gate_factory = Z


class CPHASE(ControlledGate):
    """Controlled PHASE gate."""

    gate_factory = PHASE


class SWAP(SpecializedGate):
    """Quantum SWAP gate."""

    def __init__(self, qubits: Tuple[int, int]):
        super().__init__(qubits)

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])


class XX(SpecializedGate):
    """Quantum XX gate."""

    def __init__(self, qubits: (int, int), angle: Union[float, sympy.Symbol] = sympy.Symbol("theta")):
        super().__init__(qubits)
        self.angle = angle

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([
            [sympy.cos(self.angle / 2), 0, 0, -1j * sympy.sin(self.angle / 2)],
            [0, sympy.cos(self.angle / 2), -1j * sympy.sin(self.angle / 2), 0],
            [0, -1j * sympy.sin(self.angle / 2), sympy.cos(self.angle / 2), 0],
            [-1j * sympy.sin(self.angle / 2), 0, 0, sympy.cos(self.angle / 2)]
        ])


class YY(SpecializedGate):
    """Quantum YY gate."""

    def __init__(self, qubits: (int, int), angle: Union[float, sympy.Symbol] = sympy.Symbol("theta")):
        super().__init__(qubits)
        self.angle = angle

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([
            [sympy.cos(self.angle / 2), 0, 0, 1j * sympy.sin(self.angle / 2)],
            [0, sympy.cos(self.angle / 2), -1j * sympy.sin(self.angle / 2), 0],
            [0, -1j * sympy.sin(self.angle / 2), sympy.cos(self.angle / 2), 0],
            [1j * sympy.sin(self.angle / 2), 0, 0, sympy.cos(self.angle / 2)]
        ])


class ZZ(SpecializedGate):
    """Quantum ZZ gate"""

    def __init__(self, qubits: (int, int), angle: Union[float, sympy.Symbol] = sympy.Symbol("theta")):
        super().__init__(qubits)
        self.angle = angle

    def _create_matrix(self) -> sympy.Matrix:
        arg = self.angle / 2

        return sympy.Matrix([
            [sympy.cos(arg) - 1j * sympy.sin(arg), 0, 0, 0],
            [0, sympy.cos(arg) + 1j * sympy.sin(arg), 0, 0],
            [0, 0, sympy.cos(arg) + 1j * sympy.sin(arg), 0],
            [0, 0, 0, sympy.cos(arg) - 1j * sympy.sin(arg)],
        ])
