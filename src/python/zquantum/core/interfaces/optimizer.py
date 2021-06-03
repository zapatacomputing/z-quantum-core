from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union

import numpy as np
import scipy
from scipy.optimize import OptimizeResult
from zquantum.core.interfaces.functions import CallableWithGradient
import warnings


class Optimizer(ABC):
    """
    Interface for implementing different optimizers.

    Args:
        options (dict): dictionary containing optimizer options.

    """

    def __init__(self, options: Optional[Dict] = None):
        warnings.warn(
            'Default input argument "options" will soon be removed from the '
            "optimizer interface. However, this does not preclude particular "
            'optimizers from continuing to declare "options" within their individual '
            "constructors.",
            DeprecationWarning,
        )
        if options is None:
            options = {}
        self.options = options
        if "keep_value_history" not in self.options.keys():
            self.options["keep_value_history"] = False

    @abstractmethod
    def minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        **kwargs
    ) -> OptimizeResult:
        """Finds the parameters which minimize given cost function.

        Args:
            cost_function: an object representing the cost function.
            initial_params: initial parameters for the cost function.

        Returns:
            OptimizeResults
        """
        raise NotImplementedError


def optimization_result(
    *, opt_value, opt_params, **kwargs
) -> scipy.optimize.OptimizeResult:
    """Construct instance of OptimizeResult.

    The purpose of this function is to add a safety layer by detecting if required
    components of OptimizationResult are missing already during static analysis.

    Args:
        opt_value: the final value of the function being optimized.
        opt_params: the parameters (arguments) for which opt_value has been achieved.
        kwargs: other attributes (e.g. history) that should be stored in OptimizeResult.
    Returns:
        An instance of OptimizeResult containing opt_value, opt_params and all of the
        other passed arguments.
    """
    return scipy.optimize.OptimizeResult(
        opt_value=opt_value, opt_params=opt_params, **kwargs
    )


def construct_history_info(cost_function, keep_value_history):
    histories = {
        "history": cost_function.history if keep_value_history else [],
    }

    if keep_value_history and hasattr(cost_function, "gradient"):
        histories["gradient_history"] = cost_function.gradient.history
    return histories
