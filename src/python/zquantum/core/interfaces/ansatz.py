from abc import ABC, abstractmethod
from zquantum.core.circuit import Circuit
import copy
import sympy
from typing import List


class Ansatz(ABC):

    supported_gradient_methods = ["finite_differences"]

    def __init__(
        self, n_qubits: int, n_layers: int, gradient_type: str = "finite_differences"
    ):
        """
        Interface for implementing different ansatzes.

        Args:
            n_qubits: number of qubits used for the ansatz. Note that some gradient techniques might require use of ancilla qubits.
            n_layers: number of layers of the ansatz
            gradient_type: string defining what method should be used for calculating gradients.
        
        Attributes:
            supported_gradient_methods(list): List containing what type of gradients does given ansatz support.
        
        """
        self.gradient_type = gradient_type
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self.circuit = self.get_circuit()
        self.gradient_circuits = self.get_gradient_circuits()

    def replace(self, **overrides) -> Ansatz:
        """
        Returns a new Ansatz instance with specified parameters being overriden.
        
        Args:
            overrides: keyword arguments which should be overriden in the returned object.

        Returns:
            Ansatz
        """
        new_ansatz = copy.copy(self)
        for attribute, value in overrides:
            setattr(new_ansatz, attribute, getattr(self, attribute))
        self.circuit = self.get_circuit()
        self.gradient_circuits = self.get_gradient_circuits()
        return new_ansatz

    def get_circuit(self) -> Circuit:
        """
        Returns a parametrizable circuit represention of the ansatz.

        Return:
        """

        raise NotImplementedError

    def get_gradient_circuits(self) -> List[Circuit]:
        """
        Returns a set of parametrizable circuits for calculating gradient of the ansatz.
        """
        if self.gradient_type == "finite_differences":
            return [self.get_circuit()]
        else:
            raise Exception(
                "Gradient type: {0} not supported.".format(self.gradient_type)
            )

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
