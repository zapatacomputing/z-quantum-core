from abc import ABC
from typing import Union
import sympy
import numpy as np
from ._gate import SpecializedGate


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


class X(SingleQubitGate):
    """Quantum X gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[0, 1], [1, 0]])


class Y(SingleQubitGate):
    """Quantum Y gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[0, -1.0j], [1.0j, 0.0]])


class Z(SingleQubitGate):
    """Quantum Z gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[1, 0], [0, -1]])


class H(SingleQubitGate):
    """Quantum Hadamard gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [(1 / np.sqrt(2)), (1 / np.sqrt(2))],
                [(1 / np.sqrt(2)), (-1 / np.sqrt(2))],
            ]
        )


class I(SingleQubitGate):
    """Quantum Identity gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix([[1, 0], [0, 1]])


class PHASE(SingleQubitGate):
    """Quantum Phase gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [1, 0],
                [0, complex(0, 1)],
            ]
        )


class T(SingleQubitGate):
    """Quantum T gate."""

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [1, 0],
                [0, sympy.exp(-1j * np.pi / 4)],
            ]
        )


class RX(SingleQubitGate):
    """Quantum Rx gate."""

    def __init__(
        self, qubit: int, parameter: Union[float, sympy.Symbol] = sympy.Symbol("theta")
    ):
        super().__init__(qubit)
        self.parameter = parameter

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [
                    sympy.cos(self.parameter / 2),
                    -sympy.I * sympy.sin(self.parameter / 2),
                ],
                [
                    -sympy.I * sympy.sin(self.parameter / 2),
                    sympy.cos(self.parameter / 2),
                ],
            ]
        )


class RY(SingleQubitGate):
    """Quantum Ry gate."""

    def __init__(
        self, qubit: int, parameter: Union[float, sympy.Symbol] = sympy.Symbol("theta")
    ):
        super().__init__(qubit)
        self.parameter = parameter

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [
                    sympy.cos(self.parameter / 2),
                    -1 * sympy.sin(self.parameter / 2),
                ],
                [
                    sympy.sin(self.parameter / 2),
                    sympy.cos(self.parameter / 2),
                ],
            ]
        )


class RZ(SingleQubitGate):
    """Quantum Rz gate."""

    def __init__(
        self, qubit: int, parameter: Union[float, sympy.Symbol] = sympy.Symbol("theta")
    ):
        super().__init__(qubit)
        self.parameter = parameter

    def _create_matrix(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [
                    sympy.exp(-1 * sympy.I * self.parameter / 2),
                    0,
                ],
                [
                    0,
                    sympy.exp(sympy.I * self.parameter / 2),
                ],
            ]
        )
