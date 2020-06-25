from abc import ABC, abstractmethod
from ..circuit import Circuit
from .ansatz_utils import ansatz_property
from ..utils import create_symbols_map
import copy
import sympy
from typing import List, Optional
from overrides import EnforceOverrides
import numpy as np


class Ansatz(ABC, EnforceOverrides):

    supports_parametrized_circuits = None
    n_layers = ansatz_property("n_layers")

    def __init__(self, n_layers: int):
        """
        Interface for implementing different ansatzes.
        This class also caches the circuit for given ansatz parameters.

        Args:
            n_layers: number of layers of the ansatz
        
        Attributes:
            n_layers (int): see Args
            circuit (zquantum.core.circuit.Circuit): circuit representation of the ansatz.
            supports_parametrized_circuits(bool): flag indicating whether given ansatz supports parametrized circuits.
        
        """
        self.n_layers = n_layers
        self._parametrized_circuit = None

    @property
    def parametrized_circuit(self) -> Circuit:
        if self._parametrized_circuit is None:
            if self.supports_parametrized_circuits:
                return self._generate_circuit()
            else:
                raise (
                    NotImplementedError(
                        "{0} does not support parametrized circuits.".format(
                            type(self).__name__
                        )
                    )
                )
        else:
            return self._parametrized_circuit

    def get_executable_circuit(self, parameters: np.ndarray) -> Circuit:
        """
        Returns an executable circuit representing the ansatz.
        Args:
            parameters: circuit parameters
        """
        if parameters is None:
            raise (Exception("Parameters can't be None for executable circuit."))
        if self.supports_parametrized_circuits:
            symbols = self.get_symbols()
            symbols_map = create_symbols_map(symbols, parameters)
            executable_circuit = self.parametrized_circuit.evaluate(symbols_map)
            return executable_circuit
        else:
            return self._generate_circuit(parameters)

    def _generate_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """
        Returns a circuit represention of the ansatz.
        Will return parametrized circuits if no parameters are passed and the ansatz supports parametrized circuits.
        Args:
            parameters: circuit parameters
        """
        raise NotImplementedError

    def get_symbols(self) -> List[sympy.Symbol]:
        """
        Returns a list of parameters used for creating the ansatz.
        The order of the symbols should match the order in which parameters should be passed for creating executable circuit.
        """
        raise NotImplementedError
