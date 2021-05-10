"""Module with definitions of gradient."""
import numpy as np


def finite_differences_gradient(function, finite_diff_step_size=1e-5):
    """Create a (central) finite differences gradient for a given function.

    Args:
        function: callable accepting 1-D numpy arrays and returning float.
        finite_diff_step_size: finite difference size used to estimate gradient.
    Returns:
        A function that returns a gradient estimation using central finite
        differences method.
    """

    def _gradient(parameters):
        gradient = np.array([])
        for idx in range(len(parameters)):
            values_plus = parameters.astype(float)
            values_minus = parameters.astype(float)
            values_plus[idx] += finite_diff_step_size
            values_minus[idx] -= finite_diff_step_size
            gradient = np.append(
                gradient,
                (function(values_plus) - function(values_minus))
                / (2 * finite_diff_step_size),
            )
        return gradient

    return _gradient
