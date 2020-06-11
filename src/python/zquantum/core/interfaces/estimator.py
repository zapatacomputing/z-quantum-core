from ..circuit import Circuit
from .backend import QuantumBackend
from ..measurement import ExpectationValues
from abc import ABC, abstractmethod
from openfermion import SymbolicOperator
from typing import List


class Estimator(ABC):
    """Interface for implementing different estimators.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_estimated_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
    ) -> ExpectationValues:
        """Estimators take an unknown quantum state (or the circuit that prepares the unknown state) and a list of target functions
        as input and produce a list of estimations as output. 

        Args:
            backend (QuantumBackend): The backend used to run quantum circuits, either a simulator or quantum hardware.
            circuit (Circuit): The circuit that prepares the unknown quantum state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.

        Returns:
            ExpectationValues: The estimations of the terms in the target_operator. 
        """
        raise NotImplementedError
