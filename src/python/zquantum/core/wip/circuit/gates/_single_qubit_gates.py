"""Gates acting on single qubit."""
from abc import ABC
from typing import Union, Tuple, Any
import sympy
import numpy as np
from ._gate import SpecializedGate, HermitianMixin


class SingleQubitGate(SpecializedGate, ABC):
    """Base class for single qubit gates.

    Attributes:
        qubit: index of qubit this gate acts on.
    Args:
        qubit: index of qubit this gate acts on.
    """

    def __init__(self, qubit: int):
        self.qubit = qubit
        super().__init__((qubit,))


class SingleQubitRotationGate(SingleQubitGate, ABC):
    def __init__(
        self, qubit: int, angle: Union[float, sympy.Symbol] = sympy.Symbol("theta")
    ):
        super().__init__(qubit)
        self.angle = angle

    @property
    def params(self) -> Tuple[Any, ...]:
        return (self.angle,)


class X(HermitianMixin, SingleQubitGate):
    """Quantum X gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[0, 1], [1, 0]])


class Y(HermitianMixin, SingleQubitGate):
    """Quantum Y gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[0, -1.0j], [1.0j, 0.0]])


class Z(HermitianMixin, SingleQubitGate):
    """Quantum Z gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[1, 0], [0, -1]])


class H(HermitianMixin, SingleQubitGate):
    """Quantum Hadamard gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [(1 / np.sqrt(2)), (1 / np.sqrt(2))],
                [(1 / np.sqrt(2)), (-1 / np.sqrt(2))],
            ]
        )


class I(HermitianMixin, SingleQubitGate):
    """Quantum Identity gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[1, 0], [0, 1]])


class PHASE(SingleQubitRotationGate):
    """Quantum Phase gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [1, 0],
                [0, sympy.exp(1j * self.angle)],
            ]
        )


class T(SingleQubitGate):
    """Quantum T gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [1, 0],
                [0, sympy.exp(1j * np.pi / 4)],
            ]
        )


class RX(SingleQubitRotationGate):
    """Quantum Rx gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [
                    sympy.cos(self.angle / 2),
                    -sympy.I * sympy.sin(self.angle / 2),
                ],
                [
                    -sympy.I * sympy.sin(self.angle / 2),
                    sympy.cos(self.angle / 2),
                ],
            ]
        )


class RY(SingleQubitRotationGate):
    """Quantum Ry gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [
                    sympy.cos(self.angle / 2),
                    -1 * sympy.sin(self.angle / 2),
                ],
                [
                    sympy.sin(self.angle / 2),
                    sympy.cos(self.angle / 2),
                ],
            ]
        )


class RZ(SingleQubitRotationGate):
    """Quantum Rz gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [
                    sympy.exp(-1 * sympy.I * self.angle / 2),
                    0,
                ],
                [
                    0,
                    sympy.exp(sympy.I * self.angle / 2),
                ],
            ]
        )
