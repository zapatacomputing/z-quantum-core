from abc import ABC
from typing import Tuple, Union
import sympy
from . import SpecializedGate, X, Z, PHASE, ControlledGate, HermitianMixin


class TwoQubitRotationGate(SpecializedGate, ABC):
    def __init__(
        self,
        first_qubit: int,
        second_qubit: int,
        angle: Union[float, sympy.Expr] = sympy.Symbol("theta"),
    ):
        super().__init__((first_qubit, second_qubit))
        self.angle = angle

    @property
    def params(self):
        return (self.angle,)


class CNOT(HermitianMixin, ControlledGate):
    """Controlled NOT (Controlled X) gate."""

    def __init__(self, control: int, target: int):
        super().__init__(X(target), control)


class CZ(HermitianMixin, ControlledGate):
    """"Controlled Z gate."""

    def __init__(self, control: int, target: int):
        super().__init__(Z(target), control)


class CPHASE(ControlledGate):
    """Controlled PHASE gate."""

    def __init__(
        self,
        control: int,
        target: int,
        angle: Union[float, sympy.Expr] = sympy.Symbol("theta"),
    ):
        super().__init__(PHASE(target, angle), control)
        self.angle = angle


class SWAP(HermitianMixin, SpecializedGate):
    """Quantum SWAP gate."""

    def __init__(self, first_qubit: int, second_qubit: int):
        super().__init__((first_qubit, second_qubit))

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


class XX(TwoQubitRotationGate):
    """Quantum XX gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [sympy.cos(self.angle / 2), 0, 0, -1j * sympy.sin(self.angle / 2)],
                [0, sympy.cos(self.angle / 2), -1j * sympy.sin(self.angle / 2), 0],
                [0, -1j * sympy.sin(self.angle / 2), sympy.cos(self.angle / 2), 0],
                [-1j * sympy.sin(self.angle / 2), 0, 0, sympy.cos(self.angle / 2)],
            ]
        )


class YY(TwoQubitRotationGate):
    """Quantum YY gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [sympy.cos(self.angle / 2), 0, 0, 1j * sympy.sin(self.angle / 2)],
                [0, sympy.cos(self.angle / 2), -1j * sympy.sin(self.angle / 2), 0],
                [0, -1j * sympy.sin(self.angle / 2), sympy.cos(self.angle / 2), 0],
                [1j * sympy.sin(self.angle / 2), 0, 0, sympy.cos(self.angle / 2)],
            ]
        )


class ZZ(TwoQubitRotationGate):
    """Quantum ZZ gate"""

    def _create_matrix(self) -> sympy.Matrix:
        arg = self.angle / 2

        return sympy.Matrix(
            [
                [sympy.cos(arg) - 1j * sympy.sin(arg), 0, 0, 0],
                [0, sympy.cos(arg) + 1j * sympy.sin(arg), 0, 0],
                [0, 0, sympy.cos(arg) + 1j * sympy.sin(arg), 0],
                [0, 0, 0, sympy.cos(arg) - 1j * sympy.sin(arg)],
            ]
        )
