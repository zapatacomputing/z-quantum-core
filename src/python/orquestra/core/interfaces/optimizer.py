from abc import ABC, abstractmethod
from scipy.optimize import OptimizeResult
from typing import Callable, Optional, Dict
import numpy as np

class Optimizer(ABC):
    """
    Interface for implementing different optimizers.

    Args:
        options (dict): dictionary containing optimizer options.

    """

    def __init__(self, options:Optional[Dict]=None):
        if options is None:
            options = {}
        self.options = options
        if "keep_value_history" not in self.options.keys():
            self.options["keep_value_history"] = False

    @abstractmethod
    def minimize(self, cost_function:Callable, initial_params:np.ndarray, **kwargs) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function.

        Args:
            cost_function: a cost function to be minimized, depends on some numerical parameters.
            inital_params (np.ndarray): initial parameters for the cost function

        Returns:
            OptimizeResults
        """
        raise NotImplementedError