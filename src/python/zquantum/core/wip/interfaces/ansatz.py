from typing import Optional
from abc import abstractmethod

import numpy as np
from typing_extensions import Protocol
from zquantum.core.wip import circuits


class Ansatz(Protocol):
    """Interface for implementing ansatzes."""

    @property
    @abstractmethod
    def number_of_layers(self) -> int:
        """Number of layers in the ansatz circuit."""
        # TODO: verify that docstring makes sense
        raise NotImplementedError()

    @property
    @abstractmethod
    def supports_parametrized_circuits(self) -> bool:
        """If False, calling `parametrized_circuit` should raise an exception."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def number_of_qubits(self) -> int:
        """Number of qubits in the ansatz circuit."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def number_of_params(self) -> int:
        """Number of symbols in the ansatz circuit."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def parametrized_circuit(self) -> circuits.Circuit:
        """Parametrized circuit instance for this ansatz."""
        raise NotImplementedError()

    @abstractmethod
    def generate_circuit(self, params: Optional[np.ndarray] = None) -> circuits.Circuit:
        """Circuit instance for this ansatz.

        Args:
            params: 1D array of parameters used to specialize the ansatz circuit.
                Should contain the same number of elements as the circuit's symbols.
                The params order should match the order of symbols in the circuit.

        Returns:
            Circuit with `params` applied to it.
            If `params` is `None` and the ansatz supports parametrized circuits, this
            should return a parametrized circuit.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_executable_circuit(self, params: np.ndarray) -> circuits.Circuit:
        """Circuit that can be directly executed without additional symbol substitution.

        Args:
            params: 1D array of parameters used to specialize the ansatz circuit.
                Should contain the same number of elements as the circuit's symbols.
                The params order should match the order of symbols in the circuit.
        """
        raise NotImplementedError()

    # NOTE: there's no `get_symbols() -> List[sympy.Symbol]` here.
    # We intended to get rid of it in upcoming releases anyway.
