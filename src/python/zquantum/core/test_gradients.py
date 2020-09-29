"""Tests for core.gradients module."""
import numpy as np
import pytest
from unittest.mock import Mock

from .gradients import finite_differences_gradient


def sum_x_squared(parameters: np.ndarray) -> float:
    return (parameters ** 2).sum()


@pytest.mark.parametrize(
    "parameters", [np.array([1, 2, 3]), np.array([0.5, 0.25, 1, 0.0])]
)
def test_finite_differences_gradient_returns_vectors_with_correct_length(parameters):
    gradient = finite_differences_gradient(sum_x_squared)

    assert len(parameters) == len(gradient(parameters))


@pytest.mark.parametrize(
    "epsilon,parameters",
    [
        (0.1, np.array([0, 0, 0])),
        (0.001, np.array([0, 1, 0])),
        (0.001, np.array([-0.5, 0.25, 1])),
    ],
)
def test_finite_differences_gradient_uses_supplied_epsilon_to_compute_gradient_estimate(
    epsilon, parameters
):
    gradient = finite_differences_gradient(sum_x_squared, epsilon)
    eps_vectors = np.eye(len(parameters)) * epsilon

    expected_gradient_value = np.array(
        [
            sum_x_squared(parameters + vector) - sum_x_squared(parameters - vector)
            for vector in eps_vectors
        ]
    ) / (2 * epsilon)

    assert np.array_equal(expected_gradient_value, gradient(parameters))
