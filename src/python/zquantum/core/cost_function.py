from .interfaces.cost_function import CostFunction
from .interfaces.backend import QuantumBackend
from .interfaces.ansatz import Ansatz
from .circuit import build_ansatz_circuit
from .utils import create_symbols_map
from typing import Callable, Optional, Dict
import numpy as np
import copy
from openfermion import SymbolicOperator


class BasicCostFunction(CostFunction):
    """
    Basic implementation of the CostFunction interface.
    It allows to pass any function (and gradient) when initialized.

    Args:
        function (Callable): function we want to use as our cost function. Should take a numpy array as input and return a single number.
        gradient_function (Callable): function used to calculate gradients. Optional.
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        epsilon (float): epsilon used for calculating gradient using finite difference method.

    Params:
        function (Callable): see Args
        gradient_function (Callable): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        gradient_type (str): see Args
        save_evaluation_history (bool): see Args
        epsilon (float): see Args

    """

    def __init__(
        self,
        function: Callable,
        gradient_function: Optional[Callable] = None,
        gradient_type: str = "custom",
        save_evaluation_history: bool = True,
        epsilon: float = 1e-5,
    ):
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.function = function
        self.gradient_function = gradient_function
        self.epsilon = epsilon

    def _evaluate(self, parameters: np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        value = self.function(parameters)
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


class EvaluateOperatorCostFunction(CostFunction):
    """
    Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator (openfermion.QubitOperator): operator to be evaluated
        ansatz (zquantum.core.interfaces.ansatz.Ansatz): ansatz usef to evaluate cost function
        backend (zquantum.core.interfaces.backend.QuantumBackend): backend used for evaluation
        gradient_type (str): parameter indicating which type of gradient should be used.
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        epsilon (float): epsilon used for calculating gradient using finite difference method.

    Params:
        target_operator (openfermion.QubitOperator): see Args
        ansatz (zquantum.core.interfaces.ansatz.Ansatz): see Args
        backend (zquantum.core.interfaces.backend.QuantumBackend): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        gradient_type (str): see Args
        epsilon (float): see Args

    """

    def __init__(
        self,
        target_operator: SymbolicOperator,
        ansatz: Ansatz,
        backend: QuantumBackend,
        gradient_type: str = "finite_difference",
        save_evaluation_history: bool = True,
        epsilon: float = 1e-5,
    ):
        self.target_operator = target_operator
        self.ansatz = ansatz
        self.backend = backend
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.epsilon = epsilon

    def _evaluate(self, parameters: np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            value: cost function value for given parameters, either int or float.
        """

        circuit = self.ansatz.get_executable_circuit(parameters)

        expectation_values = self.backend.get_expectation_values(
            circuit, self.target_operator
        )
        final_value = np.sum(expectation_values.values)
        return final_value
