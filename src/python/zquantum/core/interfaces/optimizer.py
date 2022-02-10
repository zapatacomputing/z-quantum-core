from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union, cast

import numpy as np
import scipy
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.cost_function import CostFunction
from zquantum.core.interfaces.functions import CallableWithGradient

from ..typing import AnyHistory, AnyRecorder, RecorderFactory


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
        cost_function: CostFunction,
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
        cost_function: CostFunction,
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


def construct_history_info(
    cost_function: AnyRecorder, keep_history: bool
) -> Dict[str, AnyHistory]:
    histories: Dict[str, AnyHistory] = {
        "history": cost_function.history if keep_history else cast(AnyHistory, []),
    }

    if keep_history and hasattr(cost_function, "gradient"):
        histories["gradient_history"] = cost_function.gradient.history
    return histories


def extend_histories(
    cost_function: AnyRecorder, histories: Dict[str, List]
) -> Dict[str, List]:
    new_histories = construct_history_info(cost_function, True)
    updated_histories = {"history": histories["history"] + new_histories["history"]}
    if hasattr(cost_function, "gradient"):
        updated_histories["gradient_history"] = (
            histories["gradient_history"] + new_histories["gradient_history"]
        )
    return updated_histories


class NestedOptimizer(ABC):
    """
    Optimizers that modify cost function throughout optimization.
    An example of such optimizer could be on that freezes certain
    parameters during every iteration or adds new layers of
    the underlying circuit (so called layer-by-layer optimization).

    See MockNestedOptimizer in zquantum.core.interfaces.mock_objects for an example.

    Args:
        inner_optimizer: Optimizer object used in the inner optimization loop.
        recorder: recorder object which defines how to store the optimization history.

    Returns:
        An instance of OptimizeResult containing:
            opt_value,
            opt_params,
            nit: total number of iterations of inner_optimizer,
            nfev: total number of calls to cost function,
            history: a list of HistoryEntrys.
                If keep_history is False this should be an empty list.
            gradient_history: if the cost function is a FunctionWithGradient,
                this should be a list of HistoryEntrys representing
                previous calls to the gradient.
    """

    @property
    @abstractmethod
    def inner_optimizer(self) -> Optimizer:
        """Inner optimizer used by this optimizer."""

    @property
    @abstractmethod
    def recorder(self) -> RecorderFactory:
        """Factory for creating recorders of functions being minimized."""
        return _recorder

    def minimize(
        self,
        cost_function_factory: Callable[..., CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Finds optimal parameters to minimize the cost function factory.

        Args:
            cost_function_factory: function that generates CostFunction objects.
            initial_params: initial parameters used for optimization
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.
        """
        return self._minimize(cost_function_factory, initial_params, keep_history)

    @abstractmethod
    def _minimize(
        self,
        cost_function_factory: Callable[..., CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """Finds the parameters which minimize given cost function factory.
        This private method should contain the integration with specific optimizer.

        Args:
            Same as for minimize.
        """
