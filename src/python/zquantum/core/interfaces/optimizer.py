from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import numpy as np
import scipy
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.cost_function import CostFunction, EstimationTasksFactory
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
        cost_function = self._preprocess_cost_function(cost_function)
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

    def _preprocess_cost_function(
        self, cost_function: Union[CallableWithGradient, Callable]
    ) -> Union[CallableWithGradient, Callable]:
        """Preprocess cost function before minimizing it.

        This method can be overridden to add some optimizer-specific features
        to cost function. For instance, an optimizer can ensure that the
        cost function has gradient, supplying a default for functions
        without one.

        Args:
            cost_function: a cost function to be preprocessed. Implementers of this
                method shouldn't mutate it.
        Returns:
            preprocess cost function, with the same signature as the original one.
        """
        return cost_function


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


class MetaOptimizer(ABC):
    def __init__(
        self,
        inner_optimizer: Optimizer,
        cost_function_factory: Callable[..., CostFunction],
        recorder: RecorderFactory = _recorder,
    ) -> None:
        """
        Optimizers that modify cost function throughout optimization.
        See RQAOA (in zquantum.qaoa) or LayerwiseAnsatzOptimizer (in
            zquantum.optimizers) for an example.

        Args:
            inner_optimizer: Optimizer object used for optimization.
            cost_function_factory: function that generates CostFunction objects.
        Returns:
            An instance of OptimizeResult containing opt_value, opt_params and other
            passed arguments.
        """
        self.inner_optimizer = inner_optimizer
        self.cost_function_factory = cost_function_factory
        self.recorder = recorder

    @abstractmethod
    def minimize(
        self,
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Finds optimal parameters to minimize the cost function.

        Args:
            initial_params: initial parameters used for optimization
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.
        """
        raise NotImplementedError
