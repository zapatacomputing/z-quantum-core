"""Class hierarchy for base gates."""
import math
from dataclasses import dataclass, field
from numbers import Number
from typing import Tuple, Union, Callable, Optional

import sympy

Parameter = Union[sympy.Symbol, Number]


@dataclass(frozen=True)
class Gate:
    """Quantum gate defined with a matrix.

    Args:
        name: Name of this gate. Implementers of new gates should make sure that the names are
            unique.
        matrix: Unitary matrix defining action of this gate.
    """

    name: str
    matrix: sympy.Matrix


def _n_qubits_for_matrix(matrix_shape):
    n_qubits = math.floor(math.log2(matrix_shape[0]))
    if 2 ** n_qubits != matrix_shape[0] or 2 ** n_qubits != matrix_shape[1]:
        raise ValueError("Gate's matrix has to be square with dimension 2^N")

    return n_qubits


def make_one_param_gate_factory(
    name: str,
    matrix_factory
):
    def _gate_factory(param):
        return Gate(
            name=name,
            matrix=matrix_factory(param)
        )

    return _gate_factory
