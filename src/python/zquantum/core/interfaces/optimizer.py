import warnings
from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import scipy
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.functions import CallableWithGradient

from ..typing import RecorderFactory


class Optimizer(ABC):
    """
    Interface for implementing different optimizers.

    Args:
        recorder: recorder object which defines how to store the optimization history.

    """

    def __init__(self, recorder: RecorderFactory = _recorder) -> None:
        self.recorder = recorder

    def minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Finds the parameters which minimize given cost function.

        Args:
            cost_function: an object representing the cost function.
            initial_params: initial parameters for the cost function.
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.
        """
        if keep_history:
            cost_function = self.recorder(cost_function)
        return self._minimize(cost_function, initial_params, keep_history)

    @abstractmethod
    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Finds the parameters which minimize given cost function.
        This private method should contain the integration with specific optimizer.

        Args:
            Same as for minimize.
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


def construct_history_info(cost_function, keep_history):
    histories = {
        "history": cost_function.history if keep_history else [],
    }

    if keep_history and hasattr(cost_function, "gradient"):
        histories["gradient_history"] = cost_function.gradient.history
    return histories
