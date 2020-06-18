from abc import ABC, abstractmethod
from ..circuit import Circuit
import copy
import sympy
from typing import List
from overrides import EnforceOverrides
from .ansatz_utils import invalidates_circuits


class Ansatz(ABC, EnforceOverrides):

    supported_gradient_methods = ["finite_differences"]

    def __init__(
        self, n_qubits: int, n_layers: int, gradient_type: str = "finite_differences"
    ):
        """
        Interface for implementing different ansatzes.
        This class also caches the circuit and gradient circuits for given ansatz parameters.

        Args:
            n_qubits: number of qubits used for the ansatz. Note that some gradient techniques might require use of ancilla qubits.
            n_layers: number of layers of the ansatz
            gradient_type: string defining what method should be used for calculating gradients.
        
        Attributes:
            n_qubits (int): see Args
            n_layers (int): see Args
            gradient_type (str): see Args
            circuit (zquantum.core.circuit.Circuit): circuit representation of the ansatz.
            gradient_circuits (List[zquantum.core.circuit.Circuit]): circuits required for calculating ansatz gradients.
            supported_gradient_methods(list): List containing what type of gradients does given ansatz support.
        
        """
        if gradient_type not in self.supported_gradient_methods:
            raise ValueError(
                "Gradient type: {0} not supported.".format(self._gradient_type)
            )
        else:
            self._gradient_type = gradient_type
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._circuit = None
        self._gradient_circuits = None

    @property
    def n_qubits(self):
        return self._n_qubits

    @invalidates_circuits
    @n_qubits.setter
    def n_qubits(self, new_n_qubits):
        self._n_qubits = new_n_qubits

    @property
    def n_layers(self):
        return self._n_layers

    @invalidates_circuits
    @n_layers.setter
    def n_layers(self, new_n_layers):
        self._n_layers = new_n_layers

    @property
    def gradient_type(self):
        return self._gradient_type

    @invalidates_circuits
    @gradient_type.setter
    def gradient_type(self, new_gradient_type):
        if new_gradient_type not in self.supported_gradient_methods:
            raise ValueError(
                "Gradient type: {0} not supported.".format(self._gradient_type)
            )
        else:
            self._gradient_type = new_gradient_type

    @property
    def circuit(self) -> Circuit:
        if self._circuit is None:
            return self.generate_circuit()
        else:
            return self._circuit

    @property
    def gradient_circuits(self) -> List[Circuit]:
        if self._circuit is None:
            return self.generate_gradient_circuits()
        else:
            return self._gradient_circuits

    def generate_circuit(self) -> Circuit:
        """
        Returns a parametrizable circuit represention of the ansatz.
        """
        raise NotImplementedError

    def generate_gradient_circuits(self) -> List[Circuit]:
        """
        Returns a set of parametrizable circuits for calculating gradient of the ansatz.
        """
        raise NotImplementedError

    def get_number_of_params_per_layer(self) -> int:
        """
        Returns number of parameters which exist in one layer of the ansatz.
        """
        raise NotImplementedError

    def get_symbols(self) -> List[sympy.Symbol]:
        """
        Returns a list of parameters used for creating the ansatz.
        """
        raise NotImplementedError
