from abc import ABC, abstractmethod
from ..circuit import Circuit
import copy
import sympy
from typing import List
from overrides import EnforceOverrides
from .ansatz_utils import invalidates_circuit


class Ansatz(ABC, EnforceOverrides):
    def __init__(self, n_qubits: int, n_layers: int):
        """
        Interface for implementing different ansatzes.
        This class also caches the circuit for given ansatz parameters.

        Args:
            n_qubits: number of qubits used for the ansatz.
            n_layers: number of layers of the ansatz
        
        Attributes:
            n_qubits (int): see Args
            n_layers (int): see Args
            circuit (zquantum.core.circuit.Circuit): circuit representation of the ansatz.
        
        """
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._circuit = None

    @property
    def n_qubits(self):
        return self._n_qubits

    @invalidates_circuit
    @n_qubits.setter
    def n_qubits(self, new_n_qubits):
        self._n_qubits = new_n_qubits

    @property
    def n_layers(self):
        return self._n_layers

    @invalidates_circuit
    @n_layers.setter
    def n_layers(self, new_n_layers):
        self._n_layers = new_n_layers

    @property
    def circuit(self) -> Circuit:
        if self._circuit is None:
            return self.generate_circuit()
        else:
            return self._circuit

    def generate_circuit(self) -> Circuit:
        """
        Returns a parametrizable circuit represention of the ansatz.
        """
        raise NotImplementedError

    def get_symbols(self) -> List[sympy.Symbol]:
        """
        Returns a list of parameters used for creating the ansatz.
        """
        raise NotImplementedError
