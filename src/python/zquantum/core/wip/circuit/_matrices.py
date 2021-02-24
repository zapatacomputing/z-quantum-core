"""Definition of predefined gate matrices and related utility functions."""
import sympy


def x_matrix():
    return sympy.Matrix([[0, 1], [1, 0]])


def y_matrix():
    return sympy.Matrix([[0, -1j], [1j, 0]])


def z_matrix():
    return sympy.Matrix([[1, 0], [0, -1]])


def rx_matrix(angle):
    return sympy.Matrix(
        [
            [sympy.cos(angle / 2), -1j * sympy.sin(angle / 2)],
            [-1j * sympy.sin(angle / 2), sympy.cos(angle / 2)],
        ]
    )


def ry_matrix(angle):
    return sympy.Matrix(
        [
            [
                sympy.cos(angle / 2),
                -1 * sympy.sin(angle / 2),
            ],
            [
                sympy.sin(angle / 2),
                sympy.cos(angle / 2),
            ],
        ]
    )


def rz_matrix(angle):
    return sympy.Matrix(
        [
            [
                sympy.exp(-1 * sympy.I * angle / 2),
                0,
            ],
            [
                0,
                sympy.exp(sympy.I * angle / 2),
            ],
        ]
    )
