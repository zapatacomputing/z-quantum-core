from .clipped_negative_log_likelihood import compute_clipped_negative_log_likelihood
from .jensen_shannon_divergence import  compute_jensen_shannon_divergence
from .mmd import compute_mmd
from .._bitstring_distribution import (
    BitstringDistribution,
    evaluate_distribution_distance,
)
from unittest import mock
import math
import pytest


def test_clipped_negative_log_likelihood_is_computed_correctly():
    """Clipped negative log likelihood between distributions is computed correctly."""
    target_distr = BitstringDistribution({"000": 0.5, "111": 0.5})
    measured_dist = BitstringDistribution({"000": 0.1, "111": 0.9})
    distance_measure_params = {"epsilon": 0.1}
    clipped_log_likelihood = compute_clipped_negative_log_likelihood(
        target_distr, measured_dist, distance_measure_params
    )

    assert clipped_log_likelihood == 1.203972804325936


def test_uses_epsilon_instead_of_zero_in_target_distribution():
    """Computing clipped negative log likelihood uses epsilon instead of zeros in log."""
    log_spy = mock.Mock(wraps=math.log)
    with mock.patch("zquantum.core.bitstring_distribution.math.log", log_spy):
        target_distr = BitstringDistribution({"000": 0.5, "111": 0.4, "010": 0.0})
        measured_dist = BitstringDistribution({"000": 0.1, "111": 0.9, "010": 0.0})
        distance_measure_params = {"epsilon": 0.01}
        compute_clipped_negative_log_likelihood(
            target_distr, measured_dist, distance_measure_params
        )

        log_spy.assert_has_calls(
            [mock.call(0.1), mock.call(0.9), mock.call(0.01)], any_order=True
        )


@pytest.mark.parametrize(
    "measured_dist,distance_measure_params,expected_mmd",
    [
        (
            BitstringDistribution({"000": 0.1, "111": 0.9}),
            {"sigma": 0.5},
            0.32000000000000006,
        ),
        (BitstringDistribution({"000": 0.5, "111": 0.5}), {"sigma": 1}, 0.00,),
        (
            BitstringDistribution({"000": 0.5, "111": 0.5}),
            {"sigma": [1, 0.5, 2]},
            0.00,
        ),
    ],
)
def test_gaussian_mmd_is_computed_correctly(
    measured_dist, distance_measure_params, expected_mmd
):
    """Maximum mean discrepancy (MMD) with gaussian kernel between distributions is computed correctly."""
    target_distr = BitstringDistribution({"000": 0.5, "111": 0.5})
    mmd = compute_mmd(target_distr, measured_dist, distance_measure_params)

    assert mmd == expected_mmd


@pytest.mark.parametrize(
    "distance_measure_function, expected_default_values",
    [
        (compute_mmd, {"sigma": 1.0},),
        (compute_clipped_negative_log_likelihood, {"epsilon": 1e-9}),
        (compute_jensen_shannon_divergence, {"epsilon": 1e-9}),
    ],
)
def test_distance_measure_default_parameters_are_set_correctly(
    distance_measure_function, expected_default_values
):
    """Default values of distance measure parameters are set correctly."""
    target_distr = BitstringDistribution({"000": 0.5, "111": 0.5})
    measured_distr = BitstringDistribution({"000": 0.1, "111": 0.9})
    distance = distance_measure_function(target_distr, measured_distr, {})
    expected_distance = distance_measure_function(
        target_distr, measured_distr, expected_default_values
    )

    assert distance == expected_distance


@pytest.mark.parametrize(
    "target_cls,measured_cls,distance_measure",
    [
        (BitstringDistribution, dict, compute_clipped_negative_log_likelihood),
        (dict, BitstringDistribution, compute_clipped_negative_log_likelihood),
        (dict, dict, compute_clipped_negative_log_likelihood),
        (BitstringDistribution, dict, compute_mmd),
        (dict, BitstringDistribution, compute_mmd),
        (dict, dict, compute_mmd),
        (BitstringDistribution, dict, compute_jensen_shannon_divergence),
        (dict, BitstringDistribution, compute_jensen_shannon_divergence),
        (dict, dict, compute_jensen_shannon_divergence),
    ],
)
def test_distribution_distance_can_be_evaluated_only_for_bitstring_distributions(
    target_cls, measured_cls, distance_measure
):
    """Distribution distance can be evaluated only if both arguments are bitstring distributions."""
    target = target_cls({"0": 10, "1": 5})
    measured = measured_cls({"0": 10, "1": 5})

    with pytest.raises(TypeError):
        evaluate_distribution_distance(target, measured, distance_measure)


@pytest.mark.parametrize(
    "distance_measure", [compute_clipped_negative_log_likelihood, compute_mmd, compute_jensen_shannon_divergence],
)
def test_distribution_distance_cannot_be_evaluated_if_supports_are_incompatible(
    distance_measure,
):
    """Distribution distance can be evaluated only if arguments have compatible support."""
    target = BitstringDistribution({"0": 10, "1": 5})
    measured = BitstringDistribution({"00": 10, "10": 5})

    with pytest.raises(RuntimeError):
        evaluate_distribution_distance(target, measured, distance_measure)


@pytest.mark.parametrize(
    "normalize_target,normalize_measured, distance_measure",
    [
        (True, False, compute_clipped_negative_log_likelihood),
        (False, True, compute_clipped_negative_log_likelihood),
        (True, False, compute_mmd),
        (False, True, compute_mmd),
        (True, False, compute_jensen_shannon_divergence),
        (False, True, compute_jensen_shannon_divergence),
    ],
)
def test_distribution_distance_cannot_be_computed_if_distributions_differ_in_normalization(
    normalize_target, normalize_measured, distance_measure
):
    """Distribution distance cannot be computed if only one distribution is normalized."""
    target = BitstringDistribution({"0": 10, "1": 5}, normalize_target)
    measured = BitstringDistribution({"0": 10, "1": 5}, normalize_measured)

    with pytest.raises(RuntimeError):
        evaluate_distribution_distance(target, measured, distance_measure)

def test_jensen_shannon_divergence_is_computed_correctly():
    """jensen shannon divergence between distributions is computed correctly."""
    target_distr = BitstringDistribution({"000": 0.5, "111": 0.5})
    measured_dist = BitstringDistribution({"000": 0.1, "111": 0.9})
    distance_measure_params = {"epsilon": 0.1}
    jensen_shannon_divergence = compute_jensen_shannon_divergence(
        target_distr, measured_dist, distance_measure_params
    )

    assert jensen_shannon_divergence == 0.9485599924429406

def test_uses_epsilon_instead_of_zero_in_target_distribution():
    """Computing jensen shannon divergence uses epsilon instead of zeros in log."""
    log_spy = mock.Mock(wraps=math.log)
    with mock.patch("zquantum.core.bitstring_distribution.math.log", log_spy):
        target_distr = BitstringDistribution({"000": 0.5, "111": 0.4, "010": 0.0})
        measured_dist = BitstringDistribution({"000": 0.1, "111": 0.9, "010": 0.0})
        distance_measure_params = {"epsilon": 0.01}
        compute_jensen_shannon_divergence(
            target_distr, measured_dist, distance_measure_params
        )

        log_spy.assert_has_calls(
            [mock.call(0.1), mock.call(0.9), mock.call(0.01)], any_order=True
        )
