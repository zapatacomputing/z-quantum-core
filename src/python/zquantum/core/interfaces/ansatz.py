from abc import ABC, abstractmethod
from zquantum.core.circuit import Circuit
import copy
import sympy
from typing import List
from overrides import EnforceOverrides


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
            supported_gradient_methods(list): List containing what type of gradients does given ansatz support.
        
        """
        if self._gradient_type not in supported_gradient_methods:
            raise ValueError(
                "Gradient type: {0} not supported.".format(self._gradient_type)
            )
        else:
            self._gradient_type = gradient_type
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._circuit = None
        self.gradient_circuits = None

    @property
    def circuit(self) -> Circuit:
        if self._circuit is None:
            return self.generate_circuit()
        else:
            return self._circuit

    @property
    def gradient_circuits(self) -> List[Circuit]:
        if self._circuit is None:
            return self.generate_circuit()
        else:
            return self._gradient_circuits

    @property
    def gradient_type(self):
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, new_gradient_type):
        if new_gradient_type not in supported_gradient_methods:
            raise ValueError(
                "Gradient type: {0} not supported.".format(self._gradient_type)
            )
        else:
            self._gradient_type = new_gradient_type
        # TODO: not sure which one is the best:
        # 1. self._regenerate_circuits()
        # 2. self._regenerate_circuits(gradients_only=True)
        # 3. self.gradient_circuits = self.generate_gradient_circuits()
        self._regenerate_circuits()

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, new_n_qubits):
        self._n_qubits = new_n_qubits
        self._regenerate_circuits()

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, new_n_qubits):
        self._n_qubits = new_n_qubits
        self._regenerate_circuits()

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

    def _regenerate_circuits(self):
        """
        Regenerates `circuit` and `gradient_circuits` after some fields of the class change.
        """
        self.circuit = self.generate_circuit()
        self.gradient_circuits = self.generate_gradient_circuits()
