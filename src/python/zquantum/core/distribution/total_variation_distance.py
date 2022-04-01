import math
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from zquantum.core.distribution import MeasurementOutcomeDistribution


def total_variation_distance(
    target_distribution: "MeasurementOutcomeDistribution",
    measured_distribution: "MeasurementOutcomeDistribution",
    distance_measure_parameters: Dict,
) -> float:
    """Compute the value of the clipped negative log likelihood between a target
     distribution and a measured distribution.
    See Equation (4) in https://advances.sciencemag.org/content/5/10/eaaw9918?rss=1

    Args:
        target_distribution: The target probability distribution.
        measured_distribution: The measured probability distribution.

    Returns:
        The value of the the total variation distance
    """

    epsilon = distance_measure_parameters.get("epsilon", 1e-9)
    value = 0.0
    target_keys = target_distribution.distribution_dict.keys()
    measured_keys = measured_distribution.distribution_dict.keys()
    all_keys = set(target_keys).union(measured_keys)

    for bitstring in all_keys:
        target_bitstring_value = target_distribution.distribution_dict.get(bitstring,0)
        measured_bitstring_value = measured_distribution.distribution_dict.get(bitstring,0)

        value += abs(target_bitstring_value - measured_bitstring_value)

    return value
