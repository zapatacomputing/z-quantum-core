from typing import Tuple, Union, Dict, TextIO
import sympy
import warnings
import numpy as np
from ._gate import Gate

# Definitely Include:
class CNOT(Gate):
    """Quantum CNOT gate"""

    def __init__(self, control: int, target: int):
        pass


class CZ(Gate):
    """Quantum CZ gate"""

    def __init__(self, control: int, target: int):
        pass


class CPHASE(Gate):
    """Quantum CPHASE gate"""

    def __init__(self, control: int, target: int):
        pass


class SWAP(Gate):
    """Quantum SWAP gate"""

    def __init__(self, qubits: (int, int)):
        pass


# Maybe Include:
class XX(Gate):
    """Quantum XX gate"""

    def __init__(self, qubits: (int, int)):
        pass


class YY(Gate):
    """Quantum YY gate"""

    def __init__(self, qubits: (int, int)):
        pass


class ZZ(Gate):
    """Quantum ZZ gate"""

    def __init__(self, qubits: (int, int)):
        pass