"""Definition of predefined gate matrices and related utility functions."""
import sympy
import numpy as np

# --- non-parametric gates ---


def x_matrix():
    return sympy.Matrix([[0, 1], [1, 0]])


def y_matrix():
    return sympy.Matrix([[0, -1j], [1j, 0]])


def z_matrix():
    return sympy.Matrix([[1, 0], [0, -1]])


def h_matrix():
    return sympy.Matrix(
        [
            [(1 / np.sqrt(2)), (1 / np.sqrt(2))],
            [(1 / np.sqrt(2)), (-1 / np.sqrt(2))],
        ]
    )


def i_matrix():
    return sympy.Matrix([[1, 0], [0, 1]])


def s_matrix():
    return sympy.Matrix(
        [
            [1, 0],
            [0, 1j],
        ]
    )


def t_matrix():
    return sympy.Matrix(
        [
            [1, 0],
            [0, sympy.exp(1j * np.pi / 4)],
        ]
    )


# --- gates with a single param ---


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


def phase_matrix(angle):
    return sympy.Matrix(
        [
            [1, 0],
            [0, sympy.exp(1j * angle)],
        ]
    )


# --- non-parametric two qubit gates ---


def cnot_matrix():
    return sympy.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )


def cz_matrix():
    return sympy.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ]
    )


def swap_matrix():
    return sympy.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def iswap_matrix():
    return sympy.Matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])


# --- parametric two qubit gates ---


def cphase_matrix(angle):
    return sympy.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, sympy.exp(1j * angle)],
        ]
    )


def xx_matrix(angle):
    return sympy.Matrix(
        [
            [sympy.cos(angle / 2), 0, 0, -1j * sympy.sin(angle / 2)],
            [0, sympy.cos(angle / 2), -1j * sympy.sin(angle / 2), 0],
            [0, -1j * sympy.sin(angle / 2), sympy.cos(angle / 2), 0],
            [-1j * sympy.sin(angle / 2), 0, 0, sympy.cos(angle / 2)],
        ]
    )


def yy_matrix(angle):
    return sympy.Matrix(
        [
            [sympy.cos(angle / 2), 0, 0, 1j * sympy.sin(angle / 2)],
            [0, sympy.cos(angle / 2), -1j * sympy.sin(angle / 2), 0],
            [0, -1j * sympy.sin(angle / 2), sympy.cos(angle / 2), 0],
            [1j * sympy.sin(angle / 2), 0, 0, sympy.cos(angle / 2)],
        ]
    )


def zz_matrix(angle):
    arg = angle / 2
    return sympy.Matrix(
        [
            [sympy.cos(arg) - 1j * sympy.sin(arg), 0, 0, 0],
            [0, sympy.cos(arg) + 1j * sympy.sin(arg), 0, 0],
            [0, 0, sympy.cos(arg) + 1j * sympy.sin(arg), 0],
            [0, 0, 0, sympy.cos(arg) - 1j * sympy.sin(arg)],
        ]
    )


def xy_matrix(angle):
    return sympy.Matrix(
        [
            [1, 0, 0, 0],
            [0, sympy.cos(angle / 2), 1j * sympy.sin(angle / 2), 0],
            [0, 1j * sympy.sin(angle / 2), sympy.cos(angle / 2), 0],
            [0, 0, 0, 1],
        ]
    )
