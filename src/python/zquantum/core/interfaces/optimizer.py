from abc import ABC, abstractmethod
from scipy.optimize import OptimizeResult
from .cost_function import CostFunction
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
    def minimize(self, cost_function:CostFunction, initial_params:np.ndarray, **kwargs) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function.

        Args:
            cost_function (zquantu.core.interfaces.CostFunction): an object representing the cost function.
            inital_params (np.ndarray): initial parameters for the cost function

        Returns:
            OptimizeResults
        """
        raise NotImplementedError