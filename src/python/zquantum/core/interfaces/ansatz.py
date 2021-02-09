from abc import ABC, abstractmethod
import numpy as np
import copy
import sympy
from typing import List, Optional
from overrides import EnforceOverrides
import warnings
from ..circuit import Circuit
from .ansatz_utils import ansatz_property
from ..utils import create_symbols_map


class Ansatz(ABC, EnforceOverrides):

    supports_parametrized_circuits = None
    number_of_layers = ansatz_property("number_of_layers")

    def __init__(self, number_of_layers: int):
        """Interface for implementing different ansatzes.
        This class also caches the circuit for given ansatz parameters.

        Args:
            number_of_layers: number of layers of the ansatz

        Attributes:
            number_of_layers (int): see Args
            parametrized_circuit (zquantum.core.circuit.Circuit): parametrized circuit representation of the ansatz. Might not be supported for given ansatz, see supports_parametrized_circuits.
            supports_parametrized_circuits(bool): flag indicating whether given ansatz supports parametrized circuits.

        """
        if number_of_layers < 0:
            raise ValueError("number_of_layers must be non-negative.")
        self.number_of_layers = number_of_layers
        self._parametrized_circuit = None

    @property
    def parametrized_circuit(self) -> Circuit:
        """Returns a parametrized circuit if given ansatz supports it."""
        if self._parametrized_circuit is None:
            if self.supports_parametrized_circuits:
                self._parametrized_circuit = self._generate_circuit()
            else:
                raise (
                    NotImplementedError(
                        "{0} does not support parametrized circuits.".format(
                            type(self).__name__
                        )
                    )
                )
        return self._parametrized_circuit

    @property
    def number_of_qubits(self) -> int:
        """Returns number of qubits ansatz circuit uses"""
        raise NotImplementedError

    @property
    def number_of_params(self) -> int:
        """Returns number of parameters in the ansatz."""

        if self.supports_parametrized_circuits:
            return len(self.parametrized_circuit.symbolic_params)
        else:
            raise NotImplementedError

    def get_executable_circuit(self, params: np.ndarray) -> Circuit:
        """Returns an executable circuit representing the ansatz.
        Args:
            params: circuit parameters
        """
        if params is None:
            raise (Exception("Parameters can't be None for executable circuit."))
        if self.supports_parametrized_circuits:
            symbols = self.parametrized_circuit.symbolic_params
            symbols_map = create_symbols_map(symbols, params)
            executable_circuit = self.parametrized_circuit.evaluate(symbols_map)
            return executable_circuit
        else:
            return self._generate_circuit(params)

    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """Returns a circuit represention of the ansatz.
        Will return parametrized circuits if no parameters are passed and the ansatz supports parametrized circuits.
        Args:
            params: circuit params
        """
        raise NotImplementedError

    def get_symbols(self) -> List[sympy.Symbol]:
        """Returns a list of symbolic parameters used for creating the ansatz."""
        warnings.warn(
            """`Ansatz.get_symbols()` will be deprecated in future releases of z-quantumcore.\
                Please use `self.parametrized_circuit.symbolic_params` instead.""",
        )
        return self.parametrized_circuit.symbolic_params
