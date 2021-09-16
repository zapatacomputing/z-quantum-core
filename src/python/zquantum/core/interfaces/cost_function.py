"""Interfaces related to cost functions."""
from typing import List, Union

import numpy as np
from typing_extensions import Protocol
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.interfaces.functions import (
    CallableStoringArtifacts,
    CallableWithGradient,
    CallableWithGradientStoringArtifacts,
)
from zquantum.core.utils import ValueEstimate


class _CostFunction(Protocol):
    """Cost function transforming vectors from R^n to numbers or their estimates."""

    def __call__(self, parameters: np.ndarray) -> Union[float, ValueEstimate]:
        """Compute  value of the cost function for given parameters."""
        ...


CostFunction = Union[
    _CostFunction,
    CallableWithGradient,
    CallableStoringArtifacts,
    CallableWithGradientStoringArtifacts,
]


class ParameterPreprocessor(Protocol):
    """Parameter preprocessor.

    Implementer of this protocol should create new array instead of
    modifying passed parameters in place, which can have unpredictable
    side effects.
    """

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        """Preprocess parameters."""
        ...
