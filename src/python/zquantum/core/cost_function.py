from .interfaces.cost_function import CostFunction
from .interfaces.backend import QuantumBackend
from .interfaces.estimator import Estimator
from .circuit import build_ansatz_circuit
from .estimator import BasicEstimator
from .utils import ValueEstimate
from typing import Callable, Optional, Dict
import numpy as np
import copy
from openfermion import SymbolicOperator


class BasicCostFunction(CostFunction):
    """Basic implementation of the CostFunction interface.
    It allows to pass any function (and gradient) when initialized.

    Args:
        function (Callable): function we want to use as our cost function. Should take a numpy array as input and return a single number.
        gradient_function (Callable): function used to calculate gradients. Optional.
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        accuracy(float): accuracy term used in finite difference approximation, as accuracy tends to 0, the approximation improves. 

    Params:
        function (Callable): see Args
        gradient_function (Callable): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        gradient_type (str): see Args
        save_evaluation_history (bool): see Args
        accuracy (float): see Args

    """

    def __init__(
        self,
        function: Callable,
        gradient_function: Optional[Callable] = None,
        gradient_type: str = "custom",
        save_evaluation_history: bool = True,
        accuracy: float = 1e-5,
    ):
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.function = function
        self.gradient_function = gradient_function
        self.accuracy = accuracy

    def _evaluate(self, parameters: np.ndarray) -> ValueEstimate:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters.
        """
        value = ValueEstimate(self.function(parameters))
        return value

    def get_gradient(self, parameters: np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        What method is used for calculating gradients is indicated by `self.gradient_type` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.gradient_type == "custom":
            if self.gradient_function is None:
                raise Exception("Gradient function has not been provided.")
            else:
                return self.gradient_function(parameters)
        elif self.gradient_type == "finite_difference":
            if self.gradient_function is not None:
                raise Warning(
                    "Using finite difference method for calculating gradient even though self.gradient_function is defined."
                )
            return self.get_gradients_finite_difference(parameters)
        else:
            raise Exception("Gradient type: %s is not supported", self.gradient_type)


class AnsatzBasedCostFunction(CostFunction):
    """
    Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator (openfermion.QubitOperator): operator to be evaluated
        ansatz (dict): dictionary representing the ansatz
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for evaluation
        estimator: (zquantum.core.interfaces.estimator.Estimator) = estimator used to compute expectation value of target operator 
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        accuracy(float): accuracy term used in finite difference approximation, as accuracy tends to 0, the approximation improves. 

    Params:
        target_operator (openfermion.QubitOperator): see Args
        ansatz (dict): see Args
        backend (zquantum.core.interfaces.backend.QuantumBackend): see Args
        estimator: (zquantum.core.interfaces.estimator.Estimator) = see Args 
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        gradient_type (str): see Args
        accuracy (float): see Args

    """

    def __init__(
        self,
        target_operator: SymbolicOperator,
        ansatz: Dict,
        backend: QuantumBackend,
        estimator: Estimator = None,
        gradient_type: str = "finite_difference",
        save_evaluation_history: bool = True,
        accuracy: float = 1e-5,
    ):
        self.target_operator = target_operator
        self.ansatz = ansatz
        self.backend = backend
        if estimator is None:
            self.estimator = BasicEstimator()
        else:
            self.estimator = estimator
        self.gradient_type = gradient_type
        self.save_evaluation_history = save_evaluation_history
        self.accuracy = accuracy
        self.evaluations_history = []

    def _evaluate(self, parameters: np.ndarray) -> ValueEstimate:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            value: cost function value for given parameters.
        """
        circuit = build_ansatz_circuit(self.ansatz, parameters)
        expectation_values = self.estimator.get_estimated_expectation_values(
            self.backend, circuit, self.target_operator
        )
        final_value = np.sum(expectation_values.values)
        return ValueEstimate(final_value)
