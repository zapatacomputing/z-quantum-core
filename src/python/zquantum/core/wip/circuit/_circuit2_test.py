import pytest
import numpy as np
import sympy

from ._builtin_gates import X, Y, Z, H, I, RX, RY, RZ, PHASE, T, CNOT, CZ, SWAP, ISWAP, CPHASE, XX, YY, ZZ
from ._gates import Circuit


RNG = np.random.default_rng(42)

EXAMPLE_OPERATIONS = tuple([
    *[gate(qubit_i)
      for qubit_i in [0, 1, 4]
      for gate in [X, Y, Z, H, I, T]],
    *[gate(angle)(qubit_i)
      for qubit_i in [0, 1, 4]
      for gate in [PHASE, RX, RY, RZ]
      for angle in [0, 0.1, np.pi / 5, np.pi, 2 * np.pi, sympy.Symbol("theta")]],
    *[gate(qubit1_i, qubit2_i)
      for qubit1_i, qubit2_i in [(0, 1), (1, 0), (0, 5), (4, 2)]
      for gate in [CNOT, CZ, SWAP, ISWAP]],
    *[gate(angle)(qubit1_i, qubit2_i)
      for qubit1_i, qubit2_i in [(0, 1), (1, 0), (0, 5), (4, 2)]
      for gate in [CPHASE, XX, YY, ZZ]
      for angle in [0, 0.1, np.pi / 5, np.pi, 2 * np.pi, sympy.Symbol("theta")]],
])


def test_creating_circuit_has_correct_gates():
    """The Circuit class should have the correct gates that are passed in"""
    circuit = Circuit(operations=EXAMPLE_OPERATIONS)
    assert circuit.operations == list(EXAMPLE_OPERATIONS)
