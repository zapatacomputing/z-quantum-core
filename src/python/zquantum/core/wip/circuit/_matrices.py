"""Definition of predefined gate matrices and related utility functions."""
import sympy


def x_matrix():
    return sympy.Matrix([[0, 1], [1, 0]])


def rx_matrix(angle):
    return sympy.Matrix([
        [sympy.cos(angle / 2), -1j * sympy.sin(angle / 2)],
        [-1j * sympy.sin(angle / 2), sympy.cos(angle / 2)]
    ])
