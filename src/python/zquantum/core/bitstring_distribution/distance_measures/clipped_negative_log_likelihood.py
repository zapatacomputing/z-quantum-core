import math
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from zquantum.core.bitstring_distribution import BitstringDistribution


def compute_clipped_negative_log_likelihood(
    target_distribution: "BitstringDistribution",
    measured_distribution: "BitstringDistribution",
    distance_measure_parameters: Dict,
) -> float:
    """Compute the value of the clipped negative log likelihood between a target bitstring distribution
    and a measured bitstring distribution
    See Equation (4) in https://advances.sciencemag.org/content/5/10/eaaw9918?rss=1

    Args:
        target_distribution (BitstringDistribution): The target bitstring probability distribution.
        measured_distribution (BitstringDistribution): The measured bitstring probability distribution.

        distance_measure_parameters (dict):
            - epsilon (float): The small parameter needed to regularize log computation when argument is zero. The default value is 1e-9.

    Returns:
        float: The value of the clipped negative log likelihood
    """

    epsilon = distance_measure_parameters.get("epsilon", 1e-9)
    value = 0.0
    target_keys = target_distribution.distribution_dict.keys()
    measured_keys = measured_distribution.distribution_dict.keys()
    all_keys = set(target_keys).union(measured_keys)

    for bitstring in all_keys:
        target_bitstring_value = target_distribution.distribution_dict.get(bitstring, 0)
        measured_bitstring_value = measured_distribution.distribution_dict.get(
            bitstring, 0
        )

        value += target_bitstring_value * math.log(
            max(epsilon, measured_bitstring_value)
        )

    return -value
