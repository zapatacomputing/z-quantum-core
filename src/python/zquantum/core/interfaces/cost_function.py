
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class CostFunction(ABC):
    """
    Interface for implementing different cost functions.

    Args:
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        gradient_type (str): parameter indicating which type of gradient should be used.
        epsilon (float): epsilon used for calculating gradient using finite difference method.

    Params:
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        gradient_type (str): see Args
        epsilon (float): see Args
    """

    def __init__(self, gradient_type:str='finite_difference', save_evaluation_history:bool=True, epsilon:float=1e-5):
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.gradient_type = gradient_type
        self.epsilon = epsilon

    
    def evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters and saves the results (if specified).
        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        value = self._evaluate(parameters)
        if self.save_evaluation_history:
            self.evaluations_history.append({'value':value, 'params': parameters})
        return value
    
    @abstractmethod
    def _evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur

        Returns:
            value: cost function value for given parameters, either int or float.
        """
        raise NotImplementedError

    def get_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        What method is used for calculating gradients is indicated by `self.gradient_type` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.gradient_type == "finite_difference":
            return self.get_gradients_finite_difference(parameters)
        else:
            raise Exception("Gradient type: %s is not supported", self.gradient_type)

    def get_gradients_finite_difference(self, parameters:np.ndarray, epsilon:Optional[float]=None) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters using finite differences method.

        Args:
            parameters (np.ndarray): parameters for which we calculate the gradient.
            epsilon (float): difference used for calculating finite differences. If not defined uses value specified by self.epsilon. Otherwise – overrides it.

        Returns:
            np.ndarray: gradient vector
        """
        if epsilon is None:
            epsilon = self.epsilon
        gradient = np.array([])
        for idx in range(len(parameters)):
            values_plus = parameters
            values_minus = parameters
            values_plus = parameters[idx] + epsilon
            values_minus = parameters[idx] - epsilon
            gradient = np.append(gradient, (self.evaluate(values_plus) - self.evaluate(values_minus)) / (2*epsilon))
        return gradient
