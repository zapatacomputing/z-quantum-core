import logging
from ..circuit import Circuit
from .backend import QuantumBackend
from ..measurement import ExpectationValues
from abc import ABC, abstractmethod
from openfermion import SymbolicOperator
from typing import Optional
from overrides import EnforceOverrides

logger = logging.getLogger(__name__)


class Estimator(ABC, EnforceOverrides):
    """Interface for implementing different estimators.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> ExpectationValues:
        """Estimators take an unknown quantum state (or the circuit that prepares the unknown state) and a list of target functions
        as input and produce a list of estimations as output. 

        Args:
            backend (QuantumBackend): The backend used to run quantum circuits, either a simulator or quantum hardware.
            circuit (Circuit): The circuit that prepares the unknown quantum state.
            target_operator (List[SymbolicOperator]): List of target functions to be estimated.
            n_samples (int): Number of measurements done. 
            epsilon (float): an error term.
            delta (float): a confidence term.

        Returns:
            ExpectationValues: The estimations of the terms in the target_operator. 
        """
        raise NotImplementedError

    @staticmethod
    def _log_ignore_parameter(
        estimator_name: str, parameter_name: str, parameter_value: float
    ):
        """In an estimator, users can specify the number of samples, an error term, and/or a confidence term. 
        Usually, selecting two terms fixes the third (e.g., chooseing a number of samples and an error term will fix the probability 
        that your estimation falls within the error bounds.) 
        
        Depending on the estimator, some of these parameters are ignored. This method is used to log when a parameter is ignored.
        See zquantum.core.estimators.BasicEstimator for an example.

        Args:
            estimator_name (str): The estimator that is ignoring the parameter.
            parameter_name (str): The parameter name, either n_samples, epsilon, or delta.
            parameter_value (float): The parameter value.
        """
        if parameter_value is not None:
            logger.warning(
                "{} = {}. {} does not use {}. The value was ignored.".format(
                    parameter_name, parameter_value, estimator_name, parameter_name,
                )
            )
