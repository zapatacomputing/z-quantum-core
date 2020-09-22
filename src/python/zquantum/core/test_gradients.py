"""Tests for core.gradients module."""
import numpy as np
import pytest

from .gradients import finite_differences_gradient


def sum_x_squared(parameters: np.ndarray) -> float:
    return (parameters ** 2).sum()


@pytest.mark.parametrize(
    "parameters", [np.array([1, 2, 3]), np.array([0.5, 0.25, 1, 0.0])]
)
def test_finite_differences_gradient_returns_vectors_with_correct_length(parameters):
    gradient = finite_differences_gradient(sum_x_squared)

    assert len(parameters) == len(gradient(parameters))
