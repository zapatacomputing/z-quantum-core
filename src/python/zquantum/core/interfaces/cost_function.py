from abc import ABC, abstractmethod
import numpy as np


class CostFunction(ABC):
    """
    Interface for implementing different cost functions.

    Args:
        save_evaluation_history (bool): flag indicating whether we want to store the history of all the evaluations.
        use_analytical_gradient (bool): flag indicating whether we want to use analytical or numerical gradient.

    Params:
        evaluations_history (list): List of the tuples (parameters, value) representing all the evaluation in a chronological order.
        save_evaluation_history (bool): see Args
        use_analytical_gradient (bool): see Args
        best_value (float): best value of the 

    """

    def __init__(self, save_evaluation_history:bool=True, use_analytical_gradient:bool=False):
        self.evaluations_history = []
        self.save_evaluation_history = save_evaluation_history
        self.use_analytical_gradient = use_analytical_gradient

    
    def evaluate(self, parameters:np.ndarray) -> float:
        """
        Evaluates the value of the cost function for given parameters.

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


    @abstractmethod
    def get_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the cost function for given parameters.
        Whether the gradient is calculated analytically (if implemented) or numerically, 
        is indicated by `self.use_analytical_gradient` field.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector 
        """
        if self.use_analytical_gradient:
            raise NotImplemented
        else:
            return self.get_numerical_gradient(parameters)
    
    @abstractmethod
    def get_numerical_gradient(self, parameters:np.ndarray) -> np.ndarray:
        """
        Evaluates the numerical gradient of the cost function for given parameters.

        Args:
            parameters: parameters for which we calculate the gradient.

        Returns:
            np.ndarray: gradient vector
        """
        raise NotImplemented