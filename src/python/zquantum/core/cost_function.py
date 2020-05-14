from .interfaces.cost_function import CostFunction
from .interfaces.backend import QuantumBackend
from .circuit import build_ansatz_circuit
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
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        use_analytical_gradient (bool): flag indicating whether we want to use analytical or numerical gradient.

    Params:
        function (Callable): see Args
        gradient_function (Callable): see Args
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        use_analytical_gradient (bool): see Args
        best_value (float): best value of the 

    """

    def __init__(self, function:Callable, gradient_function:Optional[Callable]=None, save_evaluation_history:bool=True, use_analytical_gradient:bool=False):
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.use_analytical_gradient = use_analytical_gradient
        self.function = function
        self.gradient_function = gradient_function

    def _evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        value = self.function(parameters)
        return value

    def get_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        Whether the gradient is calculated analytically (if implemented) or numerically, 
        is indicated by `use_analytical_gradient` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.use_analytical_gradient and gradient_function is not None:
            return self.gradient_function(parameters)
        else:
            return self.get_numerical_gradient(parameters)

    def get_numerical_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the numerical gradient of the cost function for given parameters.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector
        """
        gradient = []
        for idx in range(len(parameters)):
            values_plus = parameters
            values_minus = parameters
            values_plus = parameters[idx] + self.epsilon
            values_minus = parameters[idx] - self.epsilon
            gradient.append((self.evaluate(values_plus) - self.evaluate(values_minus)) / (2*self.epsilon))
        return gradient


class EvaluateOperatorCostFunction(CostFunction):
    """
    Cost function used for evaluating given operator using given ansatz.

    Args:
        target_operator
        ansatz
        backend
        constraints
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        use_analytical_gradient (bool): flag indicating whether we want to use analytical or numerical gradient.

    Params:
        target_operator
        ansatz
        backend
        constraints
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        use_analytical_gradient (bool): see Args
        best_value (float): best value of the 

    """

    def __init__(self, target_operator:SymbolicOperator, 
                        ansatz:Dict, 
                        backend:QuantumBackend, 
                        constraints:Optional[Dict]=None, 
                        save_evaluation_history:bool=True, 
                        use_analytical_gradient:bool=False):
        self.target_operator = target_operator
        self.ansatz = ansatz
        self.backend = backend
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.use_analytical_gradient = use_analytical_gradient

    def _evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        circuit = build_ansatz_circuit(self.ansatz, parameters)
        expectation_values = self.backend.get_expectation_values(circuit, self.target_operator)
        final_value = np.sum(expectation_values.values)
        return final_value

    def get_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        Whether the gradient is calculated analytically (if implemented) or numerically, 
        is indicated by `use_analytical_gradient` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.use_analytical_gradient:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_numerical_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        Whether the gradient is calculated analytically (if implemented) or numerically, 
        is indicated by `use_analytical_gradient` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        raise NotImplementedError
