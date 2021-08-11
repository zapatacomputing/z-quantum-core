"""Interfaces related to cost functions."""
import abc
from typing import List, Union

import numpy as np
from typing_extensions import Protocol
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.utils import ValueEstimate


class CostFunction(Protocol):
    """Cost function transforming vectors from R^n to numbers or their estimates."""

    @abc.abstractmethod
    def __call__(self, parameters: np.ndarray) -> Union[float, ValueEstimate]:
        """Compute  value of the cost function for given parameters."""


class EstimationTasksFactory(Protocol):
    """Factory from producing estimation tasks from R^n vectors.

    For instance, this can be used with ansatzes where produced estimation tasks
    are evaluating circuit.
    """

    @abc.abstractmethod
    def __call__(self, parameters: np.ndarray) -> List[EstimationTask]:
        """Produce estimation tasks for given parameters."""


class ParameterPreprocessor(Protocol):
    """Parameter preprocessor.

    Implementer of this protocol should create new array instead of
    modifying passed parameters in place, which can have unpredictable
    side effects.
    """

    @abc.abstractmethod
    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        """Preprocess parameters."""
