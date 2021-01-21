"""Examples of functions for which recorder works.

Note that none of this use ValueEstimate as a return value. This is because
recorders can work with callable objects of any return type.
"""
from typing import Optional
import numpy as np
from ..interfaces.functions import FunctionWithGradient, StoreArtifact


def sum_of_squares(params):
    """Multidimensional sum of squares.

    Args:
        params: numbers to be squared and summed.

    Returns:
        Sum of squares of numbers in params.
    """
    return sum(x ** 2 for x in params)


# The below implements function: f(x, y) = x ** 2  - x * y
function_1 = FunctionWithGradient(
    function=lambda params: params[0] ** 2 - params[0] * params[1],
    gradient=lambda params: np.array([2 * params[0] - params[1], -params[0]]),
)


class Function2:
    """Example of function with gradient, this time implemented as a class.

    This function is f((x, y, z)) = multiplier * (x - xy + z ** 2),
    where multiplier is fixed.

    Args:
        multiplier: fixed multiplier, see equation above.
    """

    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, params):
        """Compute value of this function.

        Args:
            params: parameters of the function. Their expected length is 3.

        Returns:
            self.multiplier * (x - xy + z ** 2) where x, y, z = params.
        """
        return self.multiplier * (params[0] - params[0] * params[1] + params[2] ** 2)

    def gradient(self, params):
        """Compute value of gradient of this function for given params.

        Args:
            params: parameters of the gradient. The expected length is 3.

        Returns:
            self.multiplier * [1 - y, -x, 2z] where x, y, z = params.
        """
        return self.multiplier * np.array([1 - params[1], -params[0], 2 * params[2]])


def function_3(params: int, store_artifact=None):
    """An example of function that stores artifacts.

    Args:
        params: parameters for the function. Note that the name `params` is
          to follow the general conversion, but the function expectes
          integer.
        store_artifact: callback for storing artifacts. See StoreArtifact
          protocol for explanation.

    Returns:
        The input argument multiplied by 2. As a side effect, binary
        representation of the input argument is stored as "bitstring"
        artifact.
    """
    bitstring = bin(params)[2:]
    if store_artifact:
        store_artifact("bitstring", bitstring)
    return 2 * params


def function_4(params: int, store_artifact: Optional[StoreArtifact] = None) -> int:
    """Another example of function that stores artifacts, but occasionally forces store.

    This function is identical to function_3, except the artifact is forcefully
    stored if input parameter is divisible by 2.
    """
    bitstring = bin(params)[2:]
    if store_artifact:
        store_artifact("bitstring", bitstring, force=params % 2 == 0)
    return 2 * params


class Function5:
    """Example of a function that stores artifacts and has gradient. Full service.

    This function is f((x, y, z)) = alpha * xyz, where alpha is fixed.

    Args:
        alpha: parameter for this function, see equation above.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, params: np.ndarray, store_artifact=None):
        """Compute value of this function.

        Args:
            params: parameters of the function. The expected length is 3.

        Returns:
            self.alpha * x * y * z, where x, y, z = params
        """
        if params[0] > 0 and store_artifact:
            store_artifact("something", params[0] + params[1])
        return self.alpha * params[0] * params[1] * params[2]

    def gradient(self, params):
        """Compute value of gradient of this function for given params.

        Args:
            params: parameters of the gradient. The expected length is 3.

        Returns:
            self.alpha * [yz, xz, xy] where x, y, z = params.
        """
        return self.alpha * np.ndarray(
            [params[1] * params[2], params[0] * params[2], params[0] * params[1]]
        )
