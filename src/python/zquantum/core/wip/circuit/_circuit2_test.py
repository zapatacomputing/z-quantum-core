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


def test_creating_circuit_has_correct_operations():
    circuit = Circuit(operations=EXAMPLE_OPERATIONS)
    assert circuit.operations == list(EXAMPLE_OPERATIONS)


def test_appending_to_circuit_yields_correct_operations():
    circuit = Circuit()
    circuit += H(0)
    circuit += CNOT(0, 2)

    assert circuit.operations == [H(0), CNOT(0, 2)]
    assert circuit.n_qubits == 3


def test_circuits_sum_yields_correct_operations():
    circuit1 = Circuit()
    circuit1 += H(0)
    circuit1 += CNOT(0, 2)

    circuit2 = Circuit([X(2), YY(sympy.Symbol("theta"))(5)])

    res_circuit = circuit1 + circuit2
    assert res_circuit.operations == [H(0), CNOT(0, 2), X(2), YY(sympy.Symbol("theta"))(5)]
    assert res_circuit.n_qubits == 6
