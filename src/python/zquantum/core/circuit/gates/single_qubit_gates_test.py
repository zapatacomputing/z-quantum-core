from .single_qubit_gates import X
import pytest
import sympy


def test_creating_X_gate():
    """The Gate class should raise an assertion error if the matrix is not square"""
    gate = X(1)
