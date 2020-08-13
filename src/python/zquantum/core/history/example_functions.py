"""Examples of functions for which recorder works.

Note that none of this use ValueEstimate as a return value. This is because
recorders can work with callable objects of any return type.
"""
from typing import Optional, Callable, Any

import numpy as np
from ..interfaces.functions import FunctionWithGradient, StoreArtifact


def sum_of_squares(params):
    """Multidimensional sum of squares."""
    return sum(x ** 2 for x in params)


# f(x, y) = x ** 2  - x * y
function_1 = FunctionWithGradient(
    function=lambda params: params[0] ** 2 - params[0] * params[1],
    gradient=lambda params: np.array([2 * params[0] - params[1], -params[0]]),
)


class Function2:
    """Another example of function with gradient, this time implemented as a class."""

    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, params):
        return self.multiplier * (params[0] - params[0] * params[1] + params[2] ** 2)

    def gradient(self, params):
        return self.multiplier * np.array([1 - params[1], -params[0], 2 * params[2]])


def function_3(n: int, store_artifact=None):
    """An example of function that stores artifacts."""
    bitstring = bin(n)[2:]
    if store_artifact:
        store_artifact("bitstring", bitstring)
    return 2 * n


def function_4(n: int, store_artifact: Optional[StoreArtifact] = None):
    """Another example of function that stores artifacts, but forces store for some condition."""
    bitstring = bin(n)[2:]
    if store_artifact:
        store_artifact("bitstring", bitstring, force=n % 2 == 0)
    return 2 * n


class Function5:
    """Example of a function that stores artifacts and has gradient. Full service."""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, params: np.ndarray, store_artifact=None):
        if params[0] > 0 and store_artifact:
            store_artifact("something", params[0] + params[1])
        return self.alpha * params[0] * params[1] * params[2]

    def gradient(self, params):
        return self.alpha * np.ndarray(
            [params[1] * params[2], params[0] * params[2], params[0] * params[1]]
        )
