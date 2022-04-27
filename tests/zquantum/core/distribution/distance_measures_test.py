################################################################################
# © Copyright 2021 Zapata Computing Inc.
################################################################################
import math
from unittest import mock

import pytest
from zquantum.core.distribution import (
    MeasurementOutcomeDistribution,
    compute_clipped_negative_log_likelihood,
    compute_jensen_shannon_divergence,
    compute_mmd,
    evaluate_distribution_distance,
    compute_total_variation_distance,
    compute_moment_based_distance
)


def test_clipped_negative_log_likelihood_is_computed_correctly():
    target_distr = MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5})
    measured_dist = MeasurementOutcomeDistribution({"000": 0.1, "111": 0.9})
    distance_measure_params = {"epsilon": 0.1}
    clipped_log_likelihood = compute_clipped_negative_log_likelihood(
        target_distr, measured_dist, distance_measure_params
    )

    assert clipped_log_likelihood == 1.203972804325936


def test_uses_epsilon_instead_of_zero_in_target_distribution():
    log_spy = mock.Mock(wraps=math.log)
    with mock.patch("zquantum.core.distribution.math.log", log_spy):
        target_distr = MeasurementOutcomeDistribution(
            {"000": 0.5, "111": 0.4, "010": 0.0}
        )
        measured_dist = MeasurementOutcomeDistribution(
            {"000": 0.1, "111": 0.9, "010": 0.0}
        )
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
            MeasurementOutcomeDistribution({"000": 0.1, "111": 0.9}),
            {"sigma": 0.5},
            0.32000000000000006,
        ),
        (
            MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5}),
            {"sigma": 1},
            0.00,
        ),
        (
            MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5}),
            {"sigma": [1, 0.5, 2]},
            0.00,
        ),
    ],
)
def test_gaussian_mmd_is_computed_correctly(
    measured_dist, distance_measure_params, expected_mmd
):
    """Maximum mean discrepancy (MMD) with gaussian kernel between distributions is
    computed correctly."""
    target_distr = MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5})
    mmd = compute_mmd(target_distr, measured_dist, distance_measure_params)

    assert mmd == expected_mmd


@pytest.mark.parametrize(
    "distance_measure_function, expected_default_values",
    [
        (
            compute_mmd,
            {"sigma": 1.0},
        ),
        (compute_clipped_negative_log_likelihood, {"epsilon": 1e-9}),
        (compute_jensen_shannon_divergence, {"epsilon": 1e-9}),
    ],
)
def test_distance_measure_default_parameters_are_set_correctly(
    distance_measure_function, expected_default_values
):
    target_distr = MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5})
    measured_distr = MeasurementOutcomeDistribution({"000": 0.1, "111": 0.9})
    distance = distance_measure_function(target_distr, measured_distr, {})
    expected_distance = distance_measure_function(
        target_distr, measured_distr, expected_default_values
    )

    assert distance == expected_distance


@pytest.mark.parametrize(
    "target_cls,measured_cls,distance_measure",
    [
        (MeasurementOutcomeDistribution, dict, compute_clipped_negative_log_likelihood),
        (dict, MeasurementOutcomeDistribution, compute_clipped_negative_log_likelihood),
        (dict, dict, compute_clipped_negative_log_likelihood),
        (MeasurementOutcomeDistribution, dict, compute_mmd),
        (dict, MeasurementOutcomeDistribution, compute_mmd),
        (dict, dict, compute_mmd),
        (MeasurementOutcomeDistribution, dict, compute_jensen_shannon_divergence),
        (dict, MeasurementOutcomeDistribution, compute_jensen_shannon_divergence),
        (dict, dict, compute_jensen_shannon_divergence),
    ],
)
def test_distribution_distance_can_be_evaluated_only_for_bitstring_distributions(
    target_cls, measured_cls, distance_measure
):
    target = target_cls({"0": 10, "1": 5})
    measured = measured_cls({"0": 10, "1": 5})

    with pytest.raises(TypeError):
        evaluate_distribution_distance(target, measured, distance_measure)


@pytest.mark.parametrize(
    "distance_measure",
    [
        compute_clipped_negative_log_likelihood,
        compute_mmd,
        compute_jensen_shannon_divergence,
    ],
)
def test_distribution_distance_cannot_be_evaluated_if_supports_are_incompatible(
    distance_measure,
):
    target = MeasurementOutcomeDistribution({"0": 10, "1": 5})
    measured = MeasurementOutcomeDistribution({"00": 10, "10": 5})

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
def test_distribution_distance_cant_be_computed_if_only_one_distribution_is_normalized(
    normalize_target, normalize_measured, distance_measure
):
    target = MeasurementOutcomeDistribution({"0": 10, "1": 5}, normalize_target)
    measured = MeasurementOutcomeDistribution({"0": 10, "1": 5}, normalize_measured)

    with pytest.raises(RuntimeError):
        evaluate_distribution_distance(target, measured, distance_measure)


def test_jensen_shannon_divergence_is_computed_correctly():
    """jensen shannon divergence between distributions is computed correctly."""
    target_distr = MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5})
    measured_dist = MeasurementOutcomeDistribution({"000": 0.1, "111": 0.9})
    distance_measure_params = {"epsilon": 0.1}
    jensen_shannon_divergence = compute_jensen_shannon_divergence(
        target_distr, measured_dist, distance_measure_params
    )

    assert jensen_shannon_divergence == 0.9485599924429406

def test_total_variation_distance_is_computed_correctly():
    target_distr = MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5})
    measured_distr = MeasurementOutcomeDistribution({"000": 0.4, "111": 0.6})
    total_variation_distance = compute_total_variation_distance(
        target_distr, measured_distr
    )

    assert total_variation_distance == 0.2

def test_moment_based_distance_is_computed_correctly():
    target_distr = MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5})
    measured_distr = MeasurementOutcomeDistribution({"000": 0.5, "111": 0.5})
    zero_moment_based_distance = compute_moment_based_distance(
        target_distr, measured_distr
    )

    target_distr = MeasurementOutcomeDistribution(
        {"000": 0.6, 
         "001": 0.2,
         "010": 0.1,
         "011": 0.1})
    measured_distr = MeasurementOutcomeDistribution(
        {"010": 0.1, 
         "011": 0.2,
         "100": 0.7})

    tvd = compute_total_variation_distance(
        target_distr, measured_distr
    )
    mbd = compute_moment_based_distance(
        target_distr, measured_distr, M=1
    )

    print('zero_momement_based_distance',zero_moment_based_distance)
    print('tvd:',tvd)
    print('mbd:',mbd)

    assert zero_moment_based_distance == 0.0 \
           and tvd == mbd