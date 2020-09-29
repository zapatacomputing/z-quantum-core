from typing import Tuple, Union, Dict, TextIO
import sympy
import warnings
import numpy as np
from ._gate import Gate


class X(Gate):
    """Quantum X gate

    Inputs:
        qubit (int): The qubit on which to apply the quantum gate
    """

    def __init__(self, qubit: int):
        matrix = sympy.Matrix([[0, 1], [1, 0]])
        super().__init__(matrix=matrix, qubits=(qubit,))


class Y(Gate):
    """Quantum Y gate

    Inputs:
        qubit (int): The qubit on which to apply the quantum gate
    """

    def __init__(self, qubit: int):
        matrix = sympy.Matrix(
            [[complex(0, 0), complex(0, -1)], [complex(0, 1), complex(0, 0)]]
        )
        super().__init__(matrix=matrix, qubits=(qubit,))


class Z(Gate):
    """Quantum Z gate

    Inputs:
        qubit (int): The qubit on which to apply the quantum gate
    """

    def __init__(self, qubit: int):
        matrix = sympy.Matrix([[1, 0], [0, -1]])
        super().__init__(matrix=matrix, qubits=(qubit,))


class H(Gate):
    """Quantum Hadamard gate

    Inputs:
        qubit (int): The qubit on which to apply the quantum gate
    """

    def __init__(self, qubit: int):
        matrix = sympy.Matrix(
            [
                [(1 / np.sqrt(2)), (1 / np.sqrt(2))],
                [(1 / np.sqrt(2)), (-1 / np.sqrt(2))],
            ]
        )
        super().__init__(matrix=matrix, qubits=(qubit,))


class I(Gate):
    """Quantum Identity gate

    Inputs:
        qubit (int): The qubit on which to apply the quantum gate
    """

    def __init__(self, qubit: int):
        matrix = sympy.Matrix([[1, 0], [0, 1]])
        super().__init__(matrix=matrix, qubits=(qubit,))


class Phase(Gate):
    """Quantum Phase gate

    Inputs:
        qubits (tuple[int]): A list of qubit indices that the operator acts on
        parameter (Union[float, sympy.Symbol]): The value of the parameter used in the Phase Gate. If a float is
            passed, the returned Gate is evaluated to the specified parameter value
    """

    def __init__(self, qubit: int):
        matrix = sympy.Matrix(
            [
                [1, 0],
                [0, complex(0, 1)],
            ]
        )
        super().__init__(matrix=matrix, qubits=(qubit,))


class T(Gate):
    """Quantum T gate

    Inputs:
        qubits (tuple[int]): A list of qubit indices that the operator acts on
        parameter (Union[float, sympy.Symbol]): The value of the parameter used in the T Gate. If a float is
            passed, the returned Gate is evaluated to the specified parameter value
    """

    def __init__(self, qubit: int):
        matrix = sympy.Matrix(
            [
                [1, 0],
                [0, sympy.exp(complex(0, np.pi / 4))],
            ]
        )
        super().__init__(matrix=matrix, qubits=(qubit,))


class Rx(Gate):
    """Quantum Rx gate

    Inputs:
        qubits (tuple[int]): A list of qubit indices that the operator acts on
        parameter (Union[float, sympy.Symbol]): The value of the parameter used in the Rx Gate. If a float is
            passed, the returned Gate is evaluated to the specified parameter value
    """

    def __init__(
        self, qubit: int, parameter: Union[float, sympy.Symbol] = sympy.Symbol("theta")
    ):
        matrix = sympy.Matrix(
            [
                [
                    sympy.cos(parameter / 2),
                    -sympy.I * sympy.sin(parameter / 2),
                ],
                [
                    -sympy.I * sympy.sin(parameter / 2),
                    sympy.cos(parameter / 2),
                ],
            ]
        )
        super().__init__(matrix=matrix, qubits=(qubit,))


class Ry(Gate):
    """Quantum Ry gate

    Inputs:
        qubits (tuple[int]): A list of qubit indices that the operator acts on
        parameter (Union[float, sympy.Symbol]): The value of the parameter used in the Ry Gate. If a float is
            passed, the returned Gate is evaluated to the specified parameter value
    """

    def __init__(
        self, qubit: int, parameter: Union[float, sympy.Symbol] = sympy.Symbol("theta")
    ):
        matrix = sympy.Matrix(
            [
                [
                    sympy.cos(parameter / 2),
                    -1 * sympy.sin(parameter / 2),
                ],
                [
                    sympy.sin(parameter / 2),
                    sympy.cos(parameter / 2),
                ],
            ]
        )
        super().__init__(matrix=matrix, qubits=(qubit,))


class Rz(Gate):
    """Quantum Rz gate

    Inputs:
        qubits (tuple[int]): A list of qubit indices that the operator acts on
        parameter (Union[float, sympy.Symbol]): The value of the parameter used in the Rz Gate. If a float is
            passed, the returned Gate is evaluated to the specified parameter value
    """

    def __init__(
        self, qubit: int, parameter: Union[float, sympy.Symbol] = sympy.Symbol("theta")
    ):
        matrix = sympy.Matrix(
            [
                [
                    sympy.exp(-1 * sympy.I * parameter / 2),
                    0,
                ],
                [
                    0,
                    sympy.exp(sympy.I * parameter / 2),
                ],
            ]
        )
        super().__init__(matrix=matrix, qubits=(qubit,))